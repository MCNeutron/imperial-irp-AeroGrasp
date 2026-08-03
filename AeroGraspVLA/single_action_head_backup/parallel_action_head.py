### Import packages ###
import torch
import torch.nn as nn

# Load modules
from flow_matching import FlowmatchingActionHead # Evo1 action head implementation

### Script definitions ###
# Integer IDs for the different embodiments
LIBERO_EMBODIMENT_ID = 0
HABITATSIM_EMBODIMENT_ID = 1

### Class definition ###
# This class should expose the same public interface as FlowmatchingActionHead
class ParallelActionHead(nn.Module): # Create new PyTorch module
    # PyTorch parent class init
    def __init__(self, config):
        super().__init__()

        # Define parallel action heads, creating two independent copies of the flow matching action head
        self.nav_head = FlowmatchingActionHead(config) # NAVIGATION HEAD, for HabitatSim dataset
        self.manip_head = FlowmatchingActionHead(config) # MANIPULATION HEAD, for LIBERO dataset

        # Get output action dims (using manipulation head/LIBERO head)
        self.action_dim = self.manip_head.action_dim
    
    ### Define forward function ###
    # Inputs have shape:
    #   fused_tokens: [B, ...]
    #   state: [B, ...] (e.g. if [x,y,z,rx,ry,rz,gripper], then shape is [B,D] where D is 7)
    #   actions_gt: [B, H, D] (H is horizon, and D is action dims)
    #   embodiment_id: [B]
    # Where B is batch size
    def forward(
            self,
            fused_tokens, # VLM output features (action head converts these into robot actions)
            state=None, # Robot proprioception/state
            actions_gt = None, # Ground truth actions (used for computing training objective)
            embodiment_id=None, # ID deciding which head is used (it is the dataset/robot identifier, e.g. tensor([0,0,1,1]), where each number represents an embodiment ID)
            state_mask=None,
            action_mask = None, # Mask indicating valid action dims
    ):
        ####################
        # The forward function splits the batch according to embodiment_id
        # i.e. HabitatSim samples go through navigation/HabitatSim action head (navigation_head.forward())
        # i.e. LIBERO samples go through manipulation/LIBERO action head (manipulation_head.forward())
        #   This is because each sample comes from two different datasets (LIBERO or HabitatSim), each of which has their own embodiment_id (defined in config.yaml)
        #   The forward function then runs the data through the corresponding action head
        # The pred_velocity and noise output from the forward function for each sample gets reconstructed back into the original batch ordering and returned
        # OUTPUTS: pred_velocity, noise (same as evo1 flowmatching action head forward function outputs)
        ####################
        # If no ground truth actions, assume it is inference instead (this functionality is present in flow_matching.py's FlowmatchingActionHead.forward())
        if actions_gt is None:
            return self.get_action(
                fused_tokens = fused_tokens,
                state = state,
                embodiment_id = embodiment_id,
                action_mask = action_mask,
            )

        # IF there is no embodiment_id provided, ASSUME dataset is LIBERO (i.e. embodiment_id = 0)
        B = fused_tokens.size(0) # Get batch size
        device = fused_tokens.device # Get device
        
        if embodiment_id is None: # Default embodiment (i.e. if no embodiment is provided)
            print(f"parallel_action_head.py: No embodiment_id provided for inference with ParallelActionHead forward()", flush=True)
            embodiment_id = torch.zeros( # Everything becomes LIBERO (as ID = 0 is LIBERO). Preserves backward compatibility
                B,
                dtype = torch.long,
                device = device,
            )
        
        # Create action masks, which selects samples
        # E.g. is embodiment_id = [0,1,0,1,], then nav_mask = [False,True,False,true], and manip_mask = [True,False,True,False]
        nav_mask = embodiment_id == HABITATSIM_EMBODIMENT_ID # Form NAVIGATION mask
        manip_mask = embodiment_id == LIBERO_EMBODIMENT_ID # Form MANIPULATION mask

        # Initialise output variables for storing outputs, if it is still None (i.e. output variable is not yet used)
        # E.g. if actions_gt shape is [4,50,24], then pred_velocity = [sample0 empty, sample1 empty, sample2 empty, sample3 empty]
        # pred_velocity has the exact same shape as actions_gt
        pred_outputs = []
        noise_outputs = []
        sample_idx = []

        # Training routing logic
        # Run forward pass depending on which dataset input data is from
        if nav_mask.any(): # Check if batch contains any navigation/HabitatSim dataset samples (for navigation tasks)
            nav_pred, nav_noise = self.nav_head( # Only send HabitatSim samples into navigation action expert (by doing [nav_mask], no LIBERO samples ever enter action head)
                fused_tokens = fused_tokens[nav_mask],
                state = state[nav_mask] if state is not None else None,
                actions_gt = actions_gt[nav_mask],
                embodiment_id = torch.zeros(nav_mask.sum(), dtype=torch.long, device=device),#embodiment_id[nav_mask], # Set embodiment_id to 0 (default) WITHIN each Flowmatching action head, now that data is routed to correct head
                state_mask = state_mask[nav_mask] if state_mask is not None else None,
                action_mask = action_mask[nav_mask] if action_mask is not None else None,
            )   
            
            # Insert navigation action head outputs/predictions into ONLY corresponding navigation dataset sample indices
            pred_outputs.append(nav_pred)
            noise_outputs.append(nav_noise)
            sample_idx.append(torch.where(nav_mask)[0].to(device))

        if manip_mask.any(): # If data is from LIBERO dataset (for manipulation tasks)
            manip_pred, manip_noise = self.manip_head(
                fused_tokens = fused_tokens[manip_mask],
                state = state[manip_mask] if state is not None else None,
                actions_gt = actions_gt[manip_mask],
                embodiment_id = torch.zeros(manip_mask.sum(), dtype=torch.long, device=device),#embodiment_id[manip_mask], # Set embodiment_id to 0 (default) WITHIN each Flowmatching action head, now that data is routed to correct head
                state_mask = state_mask[manip_mask] if state_mask is not None else None,
                action_mask = action_mask[manip_mask] if action_mask is not None else None,
            )

            # Store manipulation action head outputs into corresponding manipulation dataset sample indices
            pred_outputs.append(manip_pred)
            noise_outputs.append(manip_noise)
            sample_idx.append(torch.where(manip_mask)[0].to(device))
        
        # Safety check for empty batches (otherwise torch.cat([]) will crash)
        if len(pred_outputs) == 0:
            raise ValueError("Batch contains no known embodiment IDs")
        
        ### Reconstruct outputs ###
        pred_velocity = torch.cat(pred_outputs, dim=0)
        noise = torch.cat(noise_outputs, dim=0) # Not actually needed, as noise does not need gradients (but done for consistency to match FlowmatchingActionHead)
        sample_idx = torch.cat(sample_idx)

        # Restore original batch ordering
        sort_idx = torch.argsort(sample_idx)
        pred_velocity = pred_velocity[sort_idx]
        noise = noise[sort_idx]

        # Return the predicted velocity and noise for all samples in the batch
        # Here, output is restored to the original ordering (IMPORTANT as loss function expects the prediction order to match)
        return pred_velocity, noise
        
    ### Function definition for INFERENCE/getting action ###
    def get_action(
        self,
        fused_tokens,
        state=None,
        embodiment_id=None,
        action_mask=None,
    ):
        # IF there is no embodiment_id provided, ASSUME dataset is LIBERO (i.e. embodiment_id = 0)
        B = fused_tokens.size(0) # Get batch size
        device = fused_tokens.device
        
        if embodiment_id is None:
            print(f"parallel_action_head.py: No embodiment_id provided for inference with ParallelActionHead get_action()", flush=True)
            embodiment_id = torch.zeros(
                B,
                dtype = torch.long,
                device = device,
            )
        
        # Create action masks
        nav_mask = embodiment_id == HABITATSIM_EMBODIMENT_ID # Form NAVIGATION mask
        manip_mask = embodiment_id == LIBERO_EMBODIMENT_ID # Form MANIPULATION mask

        # Initialise actions output
        # E.g. [[0,0,...], [0,0,...], [0,0,...], [0,0,...]]
        actions = torch.empty(
            B,
            self.action_dim,
            device = device,
        )

        # Inference routing logic
        # Run forward pass depending on which dataset input data is from, and fill in actions
        if nav_mask.any(): # If data is from HabitatSim dataset (for navigation tasks)
            actions[nav_mask] = self.nav_head.get_action(
                fused_tokens = fused_tokens[nav_mask],
                state = state[nav_mask] if state is not None else None,
                embodiment_id = torch.zeros(nav_mask.sum(), dtype=torch.long, device=device),#embodiment_id[nav_mask], # Set embodiment_id to 0 (default) WITHIN each Flowmatching action head, now that data is routed to correct head
                action_mask = action_mask[nav_mask] if action_mask is not None else None,
            )

        if manip_mask.any(): # If data is from LIBERO dataset (for manipulation tasks)
            actions[manip_mask] = self.manip_head.get_action(
                fused_tokens = fused_tokens[manip_mask],
                state = state[manip_mask] if state is not None else None,
                embodiment_id = torch.zeros(manip_mask.sum(), dtype=torch.long, device=device),#embodiment_id[manip_mask], # Set embodiment_id to 0 (default) WITHIN each Flowmatching action head, now that data is routed to correct head
                action_mask = action_mask[manip_mask] if action_mask is not None else None,
            )
        
        if not nav_mask.any() and not manip_mask.any():
            raise ValueError(f"Unknown embodiment IDs: {torch.unique(embodiment_id).tolist()}")
        
        # Return actions
        return actions