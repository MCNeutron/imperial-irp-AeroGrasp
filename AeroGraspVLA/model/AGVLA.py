### Import packages ###
import sys
import os
from types import SimpleNamespace
import torch
import torch.nn as nn

# Load other repo modules
from model.internvl3.internvl3_embedder import InternVL3Embedder
from model.action_head.flow_matching import FlowmatchingActionHead

### AeroGraspVLA class definition ###
class AGVLA(nn.module):

    # Class initialisation
    def __init__(self, config: dict):
        super().__init__() # Calls nn.Module.__init__(), for setting model params

        # Store configuration
        self.config = config # Store config from input
        self._device = config.get("device", "cuda") # Set device, defaulting to CUDA
        self.return_cls_only = config.get("return_cls_only", False) # Get flag for whether model outputs CLS token embedding only

        # Vision-language embedder setup
        vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B") # Get VLM name (default to InternVL3-1B)
        self.embedder = InternVL3Embedder(model_name=vlm_name, device=self._device) # Create VL model encoder

        # Set action/policy head architecture
        action_head_type = config.get("action_head", "evo1_flowmatching").lower()
        
        ### Action head setup ###
        if action_head_type == "evo1_flowmatching":
           
            # Define action dimensions
            horizon = config.get("action_horizon", config.get("horizon", 16)) # Number of predicted future steps
            per_action_dim = config.get("per_action_dim", 7) # Per action dimension (e.g. pose + gripper)
            action_dim = horizon * per_action_dim
            
            # Store derived config values, for consistent use downstream
            config["horizon"] = horizon
            config["per_action_dim"] = per_action_dim
            config["action_dim"] = action_dim
            
            # check in case inconsistent config (sanity check)
            if action_dim != horizon * per_action_dim:
                raise ValueError(f"action_dim ({action_dim}) ≠ horizon ({horizon}) × per_action_dim ({per_action_dim})")
            
            # Save key attributes for current model
            self.horizon = horizon
            self.per_action_dim = per_action_dim
            
            # Create action head
            self.action_head = FlowmatchingActionHead(config=SimpleNamespace(
                embed_dim=config.get("embed_dim", 896), # Size of VLM output features
                hidden_dim=config.get("hidden_dim", 1024), # Width of internal MLP/transformer
                action_dim=action_dim, # Full output size
                horizon=horizon, # Sequence length
                per_action_dim=per_action_dim, # Per-step action size
                state_dim=config.get("state_dim", 7), # Robot state input dim
                state_hidden_dim=config.get("state_hidden_dim", 1024), # Internally projected state dim
                num_heads=config.get("num_heads", 8), # Num transformer head
                num_layers=config.get("num_layers", 8), # Num transformer layers
                dropout=config.get("dropout", 0.0), # (Optional) transformer dropout
                num_inference_timesteps=config.get("num_inference_timesteps", 50), # Inference steps for flow matching iterative refinement
                num_categories=config.get("num_categories", 1) # Flow matching num categories (1 means continuous output vals, not discrete vals)
            )).to(self._device) # Move to device
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")