### Import packages ###
import sys
import os
from types import SimpleNamespace
from typing import List, Union, Tuple
from PIL import Image
import torch
import torch.nn as nn

# Load other repo modules
from model.internvl3_embedder import InternVL3Embedder
from model.flow_matching import FlowmatchingActionHead
from model.parallel_action_head import ParallelActionHead # ADDED for ParallelActionHead implementation

### AeroGraspVLA class definition ###
class AGVLA(nn.Module):

    ### Class initialisation ###
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
        # Original Evo-1 flowmatching action head
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
        
        ### ADDED for ParallelActionHead implementation
        # Parallel action head, each of which are Evo-1 flowmatching action head
        elif action_head_type == "parallel_action_head":
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

            # Save the horizon lengths for each action head
            nav_horizon = config.get("nav_horizon", 10)
            manip_horizon = config.get("manip_horizon", 50)
            
            # Create action head
            self.action_head = ParallelActionHead(config=SimpleNamespace(
                embed_dim=config.get("embed_dim", 896), # Size of VLM output features
                hidden_dim=config.get("hidden_dim", 1024), # Width of internal MLP/transformer
                action_dim=action_dim, # Full output size
                horizon=horizon, # Sequence length
                nav_horizon=nav_horizon, # Navigation action head horizon/sequence length
                manip_horizon=manip_horizon, # Manipulation action head horizon/sequence length
                per_action_dim=per_action_dim, # Per-step action size
                state_dim=config.get("state_dim", 7), # Robot state input dim
                state_hidden_dim=config.get("state_hidden_dim", 1024), # Internally projected state dim
                num_heads=config.get("num_heads", 8), # Num transformer head
                num_layers=config.get("num_layers", 8), # Num transformer layers
                nav_num_layers=config.get("nav_num_layers", 18), # Num navigation action head transformer layers
                manip_num_layers=config.get("manip_num_layers", 8), # Num manipulation action head transformer layers
                dropout=config.get("dropout", 0.0), # (Optional) transformer dropout
                num_inference_timesteps=config.get("num_inference_timesteps", 50), # Inference steps for flow matching iterative refinement
                num_categories=config.get("num_categories", 1) # Flow matching num categories (1 means continuous output vals, not discrete vals)
            )).to(self._device) # Move to device
        ###
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")
        
    ### Function for returning VL embeddings ###
    # Inputs are images, an image mask, and a text prompt
    # Vision-language (VL) embeddings are later fed into action head
    def get_vl_embeddings(
        self,
        images: List[Image.Image], # List of images from different camera views
        image_mask: torch.Tensor,  
        prompt: str = "",
        return_cls_only: Union[bool, None] = None
    ) -> torch.Tensor:

        # Default return value
        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        # Safety check: VLM requires at least one image
        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image (PIL.Image). Got `images=None` or empty list.")
        
        # Return encoded image + text embedding
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
        )
    
    ### Function for preparing robot state ###
    # Ensures robot states are properly shaped PyTorch tensors on the correct device
    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:

        # If not already, convert list input into a tensor
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")

        # Ensure batch dim exists (so inputs are 2D)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)
    
    ### Function for running inference or training for action model ###
    # Depending on whether actions_gt is provided, function runs inference (generate actions) or training (compute flow-matching loss) from action mdoel
    # No ground truth --> Predict actions
    # With ground truth --> Train on actions
    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        if actions_gt is None: # Inference
            return self.action_head.get_action(fused_tokens, state=state, action_mask=action_mask, embodiment_id=embodiment_ids)
        else: # Training
            return self.action_head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, embodiment_id=embodiment_ids)
        
    ### Function for full inference pipeline for robot policy ###
    @torch.no_grad() # Disable gradient tracking
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None, # ADDED for parallelActionHead implementation
        embodiment_ids = None ### ADDED for ParallelActionHead implementation
    ) -> torch.Tensor:

        # Get VL embeddings
        fused_tokens = self.get_vl_embeddings(
                        images=images,
                        image_mask=image_mask,
                        prompt=prompt,
                        return_cls_only=return_cls_only
                        
                    )

        # State preprocessing
        state_tensor = self.prepare_state(state_input)  
        
        # Predict actions
        # return self.predict_action(fused_tokens, state_tensor, action_mask=action_mask)
        return self.predict_action(fused_tokens, state_tensor, action_mask=action_mask, embodiment_ids=embodiment_ids) ### ADDED for ParallelActionHead implementation
    
    ### Function for forward pass ###
    def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None):
        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    ### Function for preventing a module from being trained ###
    # By disabling gradients
    def _freeze_module(self, module: nn.Module, name: str):
        print(f"Freezing {name} parameters...")
        for p in module.parameters():
            p.requires_grad = False

    ### Function for setting training configuration ###
    def set_finetune_flags(self):
        config = self.config  
        if not config.get("finetune_vlm", False): # If NOT freeze VLM
            self._freeze_module(self.embedder, "VLM (InternVL3)")
        else:
            print("Finetuning VLM (InternVL3)...")

        if not config.get("finetune_action_head", False): # If NOT freeze action head
            self._freeze_module(self.action_head, "Action Head")
        else:
            print("Finetuning Action Head...")