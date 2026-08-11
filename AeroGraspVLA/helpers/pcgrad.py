####################
# This script defines the PCGrad backward function for performing PCGrad during AGVLA training
# while keeping Accelerate and DeepSpeed ZeRO-2 functionality.
# So this PCGrad implementation will be implemented directly around accelerator.backward() (instead
# of a separate optimiser class)
#
# PCGrad needs NAVIGATION and MANIPULATION gradients separately to compute final gradient, instead
# doing one backward pass on the combined loss.
####################

### Import packages ###
import torch

### Definition for PCGrad backward function ###
# Perform PCGrad using Accelerate's backward() mechanism and parameter gradient hooks
# INPUTS:
#   accelerator: HuggingFace Accelerate Accelerator
#   model: Accelerate/DeepSpeed-wrapped model
#   losses: list of task losses, e.g. [nav_loss, manip_loss]
#   retain_graph: Whether to retain the computation graph between task-specific backward passes
# OUTPUTS:
#   task_grad_norms: list containing the L2 norm of each task gradient
# NOTE: This function is
#   Compatible with accelerator.backward(). By passing in accelerator object, the function allows doing
#   accelerator.backward(loss), which allows it to work with distributed/DeepSpeed setups.
#   Designed for AGVLA two-task setup - i.e. [navigation loss, manipulation loss]
#   Gradients are collected after each backward pass, then projected (for PCGrad)
#   This version does NOT rely on param.grad to extract the task gradients, as under the current DeepSpeed/ZeRO
#       configuration, param.grad is not populated in a way where parameter gradients can be read from it.
#
# This function essentially replaces the normal:
#   loss = nav_loss + manip_loss
#   accelerator.backward(loss)
#   optimizer.step()
# With:
#   1. Registers a hook on every trainable parameter
#   2. Calls accelerator.backward(nav_loss)
#   3. The hooks capture the navigation gradients
#   4. Clears the capture dictionary
#   5. Calls accelerator.backward(manip_loss, retain_graph=False)
#   6. The hooks capture the manipulation gradients
#   7. Perform PCGrad projection logic on captured gradients (check whether nav and manip gradients conflict, remove conflicting components, then add the cleaned gradients together)
#   8. Use DeepSpeed's safe_set_full_grad() to put the final PCGrad gradients back into DeepSpeed's gradient state
#
# In a normal, combined loss scenario, if the dot product of the navigation and manipulatino gradients
# are negative, it means moving in the direction preferred by one task tends to increase the other task's loss
#   PCGrad attempts to remove this conficting component before combining the gradients.
def pcgrad_backward(accelerator, model, losses, retain_graph=True):
    # Ensure there are both nav and manip losses (even if one has entry None)
    if len(losses) != 2: raise ValueError("PCGrad expects [nav_loss, manip_loss].")
    if all(loss is None for loss in losses): raise ValueError("PCGrad received no valid losses")

    # Get all trainable params (including shared VLM params and potentially navigation and manipulation action head params). Stored like params=[p0, p1, p2, ...]
    # IMPORTANT as function later manually assigns gradients back to each param
    params = [p for p in model.parameters() if p.requires_grad]

    ### Initialise gradient hook storage ###
    # Initialise empty dictionaries to store gradient hooks for each task
    # Each dictionary will contain:
    #   parameter -> gradient
    # For ONE task
    nav_grads_dict = {} # Dictionary for navigation gradient hooks
    manip_grads_dict = {} # Dictionary for manipulatino gradient hooks

    current_grads = nav_grads_dict # Initialise current gradient hooks

    ### Register parameter hooks ###
    hooks = [] # Initialise list for hooks

    # This loop essentially attaches a hook that saves an independent copy of every model param's gradient whenever PyTorch computes it during backward.
    # NOTE: In this implementation, current_grads is looked up when the hook executes, NOT when the hook is created
    #   This is what is wanted, as before the first backward: current_grads --> nav_grads_dict, and before the second backward: current_grads --> manip_grads_dict
    #   So the same hooks can capture the two different backward passes
    # Look through all trainable params
    for p in params:
        # Define a function for creating the hook
        #   Each hook here remembers which param it belongs to
        def make_hook(param):
            # Define function for defining the actual hook
            #   It saves a copy of the grad, and returns the original grad
            #   NOTE: Because current_grads is selected here, the SAME hooks can capture different tasks' gradients into different dictionaries
            def hook(grad):
                # Save an independent copy of the gradient into dictionary because the gradient tensor may subsequently be modified/reused.
                #   .detach() for removing gradient tensor from autograd graph (to prevent it from participating in another backward computation)
                #   .clone() as original grad tensor may subsequently be modified, reused, accumulated, or otherwise changed.
                current_grads[param] = grad.detach().clone()

                # Return original gradient
                #   Hook isn't supposed to replace or modify the gradient. It just observes it and makes a copy
                #   Returning grad means to continue the normal backward pass using the original gradient.
                return grad

            # Return the hook (associated with p)
            return hook

        # Register the hook
        #   Return value of register_hook() is a hook handle, which gets stored in hooks
        hooks.append(p.register_hook(make_hook(p)))

    ### Compute gradients for each task separately ###
    # Gradients for each task are vectors
    # Put the two gradient dictionaries into a list
    #   Creates a list where the index corresponds to the task (e.g. task index 0 --> nav_loss --> mav_grads_dict)
    #   Avoids need to write seperate code for nav and manip
    task_grads_dicts = [nav_grads_dict, manip_grads_dict]
    
    # Loop through each task's losses
    #   Losses should be losses = [nav_loss, manip_loss] (from input)
    #   So task_idx=0 --> loss=nav_loss, task_idx=1 --> loss=manip_loss
    for task_idx, loss in enumerate(losses):
        ### Check if there are losses in current batch ###
        # Skip tasks that don't have a loss (for cases where a batch only has data for only one task)
        if loss is None:
            continue

        # Select gradient dictionary into which hooks should write
        current_grads = task_grads_dicts[task_idx]

        ### Clear previous captured gradients in current dictionary before computing this task's gradient ###
        # Prevent accidentally using stale gradients from a previous backward pass
        current_grads.clear()

        ### Determine whether another valid loss follows ###
        # With [nav, manip]: NAV --> retain_graph=True, MANIP --> retain_graph=False
        # With [nav, None]: NAV --> retain_graph=False
        # With [None, manip]: MANIP --> retain_graph=False
        remaining_valid_losses = any(losses[k] is not None for k in range(task_idx + 1, len(losses)))

        ### Backpropagate the current task's loss ###
        # After this command, PyTorch places the gradients of the loss wrt to each param into p.grad, e.g. p0.grad = grad of nav_loss wrt p0, p1.grad = grad of nav_loss wrt p1, ...
        # NOTE: Must retain graph, as for multiple tasks (nav, manip), both losses may depend on same forward-pass computation (particularly the shared VLM)
        #   Since after backward(), PyTorch normally frees computation graph to save memory, backprop through same graph again can cause RuntimeError
        #   So set this flag to tell PyTorch to keep computation graph, as will need to perform another backward pass (for next task)
        accelerator.backward(loss, retain_graph=(retain_graph or remaining_valid_losses)) # Always retain graph if retain_graph=True, or retain it for every task except final one (as no following computation graph after last task to backprop)

        ### DEBUGGING
        # if accelerator.is_main_process:
        #     print(f"\n===== PCGrad TASK {task_idx} =====", flush=True)
        #     print("Captured gradients:", len(current_grads), flush=True)
        #     count = 0

        #     for p in params:
        #         if p not in current_grads:
        #             continue

        #         g = current_grads[p]

        #         if torch.any(g != 0):
        #             print("Gradient:", tuple(g.shape), "| norm:", g.float().norm().item(), flush=True)

        #             count += 1
        #             if count >= 3:
        #                 break
        ###

    ### Remove hooks ###
    for hook in hooks:
        hook.remove()

    ### Convert dictionaries to lists ###
    # Converts per-task gradient dictionaries into ordered lists of gradients
    # Maintain exactly the same ordering as 'params'
    #   Important for PCGrad code as later it does zip(projected_grads[i], task_grads[j]) --> Needs the grads for corresponding params to be in the same position
    task_grads = [] # Initialise task_grads as an empty list (will subsequently look like task_grads[0] --> nav grads, task_grads[1] --> manip grads)

    # Loop through each task's grad dictionary
    for grads_dict in task_grads_dicts:
        # Initialise empty list for current task
        #   Will subsequently contain grads for all params, in the exact order of params
        grads =[]

        # Loop through params in the ORIGINAL ORDER
        for p in params:
            # If current param has a gradient (i.e. if current task produced a grad for this particular param) - Matters to check as not every param necessarily participates in every task (e.g. nav loss does not participate in manip head)
            if p in grads_dict:
                grads.append(grads_dict[p]) # Append grad
            # If no gradient
            #   IMPORTANT that instead of just skipping params with no grad, it preserves the position of the param (by appending None)
            #   Maintains same length and ordering of the nav grad and manip grad lists. List indices will also correspond to the same param. (Ensures PCGrad compares grads belonging to SAME params)
            else:
                grads.append(None) # Append None

        ### Store the current task's gradients ###
        # E.g. task_grads[0] = [grad of loss wrt to p0, grad of loss wrt to p1, ...]
        #   Task grads will be: task_grads[task_idx][param_idx]
        # This ensures later, gi and gj will always correspond to the SAME MODEL PARAM, as both lists were constructed using for p in params: in exactly the same order.
        task_grads.append(grads)

    ### Initialise PCGrad diagonstics ###
    # For the two-task case: task 0 = navigation, task 1 = manipulation
    pcgrad_dot_product = None
    pcgrad_norm_nav = None
    pcgrad_norm_manip = None
    pcgrad_num_projections = 0

    ### Perform PCGrad projection ###
    # PCGrad projection:
    # For each task i:
    #   if dot(g_i, g_j) < 0:
    #       g_i <- g_i - dot(g_i, g_j) / ||g_j||^2 * g_j

    # Make an independent copy of task_grads for PCGrad projection (projected_grads will subsequently contain the modified PCGrad grads)
    projected_grads = [
        [ None if g is None else g.clone() for g in task_grad]
        for task_grad in task_grads
    ]

    # Get number of tasks (2 for nav and manip)
    num_tasks = len(task_grads)

    # Loop through all combinations of tasks, but skip comparing a task with itself
    # No need to compare same tasks, as only care about conflicts between DIFFERENT tasks
    # NOTE that this loops over both directions
    #   For two tasks, it first does i = nav, j = manip: g_nav <-- g_nav - (g_nav*g_manip)/||g_manip||^2 * g_manip
    #   Then it does i = manip, j = nav: g_manip <-- g_manip - (g_manip*g_nav)/||g_nav||^2 * g_nav
    # This allows both tasks to have their conflicting components removed.
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j: # Skip comparing a task with itself
                continue
            
            ### Compute dot product and norm of task j ###
            # IMPORTANT: With DeepSpeed ZeRO-2, gradients can be partitioned across ranks. We therefore all-reduce the scalar dot product and norm
            dot_product = None
            norm_sq = None

            # Loop through all PAIRS (NOT combinations) of parameters in the current tasks being compared
            # gi and gj are the gradients for each parameter p
            for gi, gj in zip(projected_grads[i], task_grads[j]):
                ### Skip parameters that have no gradients for the current task ###
                # This will be encountered e.g. for navigation losses that do not have gradients in the manipulation head, or manip losses that do not have gradients in the nav head
                # If task j has no gradient for this parameter, then there is nothing to project for this parameter --> SKIP
                if gj is None:
                    continue

                # If task i has no gradient for this parameter, then there is nothing to project for this parameter --> SKIP
                if gi is None:
                    continue
                
                ### Calcuate gradient dot product ###
                # Gradient dot product calculated across ALL PARAMETERS (mathematically equivalent to flattening the model gradients, and calculating the dot product of g_nav and g_manip)
                # Dot product tells whether the gradients point in similar or opposing directions
                #   A positive dot product (g_i * g_j > 0) means they are GENERALLY ALIGNED (no projection occurs)
                #   Zero (g_i * g_j = 0) means they are ORTHOGONAL, no conflict
                #   Negative (g_i * g_j < 0) means THEY CONFLICT --> PCGrad performs projection
                local_dot = torch.sum(gi * gj)

                ### Calculate norm of task j ###
                # Calculates ||gj||^2 (required by PCGrad projection formula)
                local_norm_sq = torch.sum(gj * gj)

                ### Calculate dot product of the entire task gradient vectors ###
                # Each loop does g_nav*g_manip for the current param pair. Dot product is the sum of this for all param pairs. So sum them to get final dot product
                if dot_product is None:
                    dot_product = local_dot
                    norm_sq = local_norm_sq
                else:
                    dot_product = dot_product + local_dot
                    norm_sq = norm_sq + local_norm_sq
            
            # IF no overlapping gradients
            # i.e. IF no params for which both tasks have gradients, then there is nothing to compare/project, SO skip this whole loop (and so dot_product will remain as None)
            if dot_product is None:
                continue
            
            ### Aggregate across distributed processes ###
            # IF using distributed training, aggregate (combine to form a single value representing all values, e.g. a mean) dot_product and norm_sq across distributed processes
            # Necessary as PCGrad needs the global gradient relationship, instead of just the gradient relationship on one GPU
            if accelerator.num_processes > 1:
                torch.distributed.all_reduce(dot_product, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(norm_sq, op=torch.distributed.ReduceOp.SUM)
            
            ### Record PCGrad diagonstics for NAV <-> MANIP ###
            # Only relevant for the two-task case
            if num_tasks == 2:
                if i == 0 and j == 1:
                    pcgrad_dot_product = dot_product.detach().clone()
                    pcgrad_norm_manip = torch.sqrt(norm_sq + 1e-12).detach()
                elif i == 1 and j == 0:
                    pcgrad_norm_nav = torch.sqrt(norm_sq + 1e-12).detach()

            ### Perform PCGrad projection ###
            # Standard PCGrad projection is:
            #   gi <-- gi - (gi*gj)/||gj||^2 * gj
            ## Project only if gradients conflict ##
            # Gradients only conflict when gi*gj < 0 (i.e. when dot_product.item() < 0)
            if dot_product.item() < 0 and norm_sq.item() > 0: # norm_sq.item() > 0 is a safety check for division by 0 (should never happen though, if dot_product < 0. It can only happen if dot_product = 0)
                # record that a PCGrad projection actually occurred
                pcgrad_num_projections += 1

                # Calculate projection coefficient
                #   (gi*gj)/||gj||^2
                projection_coeff = dot_product / (norm_sq + 1e-12)

                # Apply projection parameter-by-parameter
                # k represents the parameter (tensor) index, while gi and gj are the gradients of that SAME parameter for two different tasks
                for k, (gi, gj) in enumerate(zip(projected_grads[i], task_grads[j])):
                    if gi is None or gj is None:
                        continue
                    
                    # Apply the projection to each parameter, for each task.
                    # This line will place the projected gradient into the row corresponding to the task (i), and the column corresponding to the current parameter (k)
                    # NOTE that gi comes from projected_grads, but gj comes from task_grads
                    #   This means task j remains the ORIGINAL gradient, while task i is PROGRESSIVELY PROJECTED
                    #   This is INTENTIONAL
                    # After projection, initial g_nav and g_manip become projected_g_nav and projected_g_manip. These are the gradients that PCGrad will actually use
                    #   But the original gradients remain available in task_grads (allows the calculation of the original gradient norms later)
                    projected_grads[i][k] = (gi - projection_coeff * gj) # Projects away conflicting components
    
    ### Combine projected task gradients ###
    # Combine the projected grads from all tasks (projected_grads) into one gradient per model param
    #   i.e. For each param, collect all available projected task grads and add them together
    # NOTE: Standard PCGrad uses the sum of projected gradients
    # Initialise empty list to store final grads (will eventually contain one grad for every param in params, e.g. final_grads:[grad_for_param0, grad_for_param1, ...])
    final_grads = []

    for p_idx in range(len(params)): # Loop through each model parameter
        # Gather the projected gradients for current parameter (p_idx), e.g. grads_for_param = [[nav_proj_grads], [manip_proj_grads]]
        grads_for_param = [task_grad[p_idx] for task_grad in projected_grads if task_grad[p_idx] is not None]

        ### Add the projected gradients together ###
        # g_final = g_nav,PCGrad + g_manip,PCGrad
        if len(grads_for_param) == 0: # IF no task has a gradient
            final_grads.append(None) # Append None, as no task produced a grad for the current param
        else: # Stack the gradients
            combined_grad = torch.stack(grads_for_param, dim=0).sum(dim=0) # Creates a new task dim, and sum across the task dim to get the combined gradient

            ### Put final gradient into p.grad ###
            # IMPORTANT for allowing PyTorch's optimiser to see p.grad --> PCGrad-combined gradient
            # So when training loop subsequently does optimizer.step(), optimizer updates the model using the PCGrad gradient.
            # NOTE: The function itself does NOT update the model. It only prepares p.grad for the optimizer
            #   The optimizer still performs the actual update.
            final_grads.append(combined_grad) # final_grads[p_idx] should contain the final gradient that should be assigned to param p_idx

    ### Write final gradients into p.grad ###
    # IMPORTANT for allowing PyTorch's optimiser to see p.grad --> PCGrad-combined gradient
    # So when training loop subsequently does optimizer.step(), optimizer updates the model using the PCGrad gradient.
    # NOTE: The function itself does NOT update the model. It only prepares p.grad for the optimizer
    #   The optimizer still performs the actual update.
    # Pair params with final grads (works as both lists were constructed using the same params ordering)
    for p, grad in zip(params, final_grads):
        # IF current param has no grad (i.e. no task produced a grad for this param)
        if grad is None:
            # No gradient for this parameter
            p.grad = torch.zeros_like(p)
        # IF current param has grad
        else:
            p.grad = grad # Set the PCGrad-computed gradient for param p
    
    ### Calculate PCGrad cosine similarity diagonstics ###
    if num_tasks == 2 and pcgrad_dot_product is not None and pcgrad_norm_nav is not None and pcgrad_norm_manip is not None:
        cosine_similarity = pcgrad_dot_product / ((pcgrad_norm_nav * pcgrad_norm_manip) + 1e-12)
    else:
        cosine_similarity = None

    ### Return task gradient norms for optional logging ###
    # Calculate ||g||^2 for each original task gradient, then converts to ||g||
    # NOTE These are the ORIGINAL task gradient norms, NOT the norms after PCGrad projection
    task_grad_norms = []
    for task_grad in task_grads:
        local_sq_norm = None

        for g in task_grad:
            if g is None:
                continue
            
            value = torch.sum(g * g)

            if local_sq_norm is None:
                local_sq_norm = value
            else:
                local_sq_norm = local_sq_norm + value
        
        if local_sq_norm is None:
            local_sq_norm = torch.tensor(0.0, device=accelerator.device)

        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(local_sq_norm, op=torch.distributed.ReduceOp.SUM)
        
        task_grad_norms.append(torch.sqrt(local_sq_norm + 1e-12))
    
    ### Return all diagonstics ###
    return {
        "task_grad_norms": task_grad_norms, # Original task gradient norms
        "dot_product": pcgrad_dot_product, # Relationship between NAV and MANIP gradients
        "cosine_similarity": cosine_similarity, # Cosine similarity
        "num_projections": pcgrad_num_projections, # Number of actual projections performed
    }