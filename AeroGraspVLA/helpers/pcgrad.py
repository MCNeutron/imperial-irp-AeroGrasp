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
# Perform PCGrad using Accelerate's backward() mechanism
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
#   Final projected gradients are written back into p.grad
#
# This function essentially replaces the normal:
#   loss = nav_loss + manip_loss
#   accelerator.backward(loss)
#   optimizer.step()
# With:
#   1. Get navigation gradient
#   2. Get manipulation gradient
#   3. Check whether they conflict
#   4. Remove conflicting components
#   5. Add the cleaned gradients together
#   6. Put the result into p.grad
#   7. optimizer.step()
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

    # Initialise empty list to store gradients for each task
    # Will look like e.g. task_grads=[nav_grads, manip_grads], where each task's grads are a list corresponding to the model params (e.g. [grad_for_p0, grad_for_p1, ...])
    task_grads = []

    ### Compute gradients for each task separately ###
    # Gradients for each task are vectors
    # Loop through each task's losses
    for task_idx, loss in enumerate(losses):
        ### Check if there are losses in current batch ###
        # If none, skip
        if loss is None:
            task_grads.append([None] * len(params))
            continue

        ### Clear gradients before computing this task's gradient, starting with p.grad=None for every param ###
        # VERY IMPORTANT, as don't want any previous gradient sitting inside p.grad (otherwise PyTorch could accumulate gradients)
        for p in params:
            p.grad = None

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

        ### Copy current task's gradients after backward pass, because the next backward pass will overwrite them ###
        grads = []

        # Loop through each param
        for p in params:
            if p.grad is None: # IF parameter recieved no gradient from the current task (e.g. nav head shouldn't receive gradients from every task (manip tasks))
                grads.append(None)
            else: # IF param recieved a gradient from current task
                grads.append(p.grad.detach().clone()) # Save an independent copy of grad (as otherwise may end up with references to grad tensors that are subsequently modified/reused)

        ### Store the current task's gradients ###
        # E.g. task_grads[0] = [grad of loss wrt to p0, grad of loss wrt to p1, ...]
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
    # NOTE: Standard PCGrad uses the sum of projected gradients
    for p_idx, p in enumerate(params): # Loop through each model parameter
        # Gather the projected gradients for current parameter (p_idx), e.g. grads_for_param = [[nav_proj_grads], [manip_proj_grads]]
        grads_for_param = [task_grad[p_idx] for task_grad in projected_grads if task_grad[p_idx] is not None]

        ### Add the projected gradients together ###
        # g_final = g_nav,PCGrad + g_manip,PCGrad
        if len(grads_for_param) == 0:
            p.grad = None
        else:
            combined_grad = torch.stack(grads_for_param, dim=0).sum(dim=0)

            ### Put final gradient into p.grad ###
            # IMPORTANT for allowing PyTorch's optimiser to see p.grad --> PCGrad-combined gradient
            # So when training loop subsequently does optimizer.step(), optimizer updates the model using the PCGrad gradient.
            # NOTE: The function itself does NOT update the model. It only prepares p.grad for the optimizer
            #   The optimizer still performs the actual update.
            p.grad = combined_grad
    
    ### Calculate PCGrad cosine similarity diagonstics ###
    if num_tasks == 2 and pcgrad_dot_product is not None:
        cosine_similarity = pcgrad_dot_product / ((pcgrad_norm_nav * pcgrad_norm_manip) + 1e-12).detach()
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