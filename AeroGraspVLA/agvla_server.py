### Import packages ###
import sys
import os
import asyncio
import websockets
import numpy as np
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms

# Load other repo modules
from model.AGVLA import AGVLA
from helpers.Normaliser import Normalizer

### Script definitions ###
# Define server port
PORT = 9000

# Integer IDs for the different embodiments
from model.embodiment_id import LIBERO_EMBODIMENT_ID, HABITATSIM_EMBODIMENT_ID

# Define model checkpoint directory
# CKPT_DIR = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/weights/agvla_libero_evo1_weights" ### ADDED - For LIBERO
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_libero/stage1_only_spatial/step_best" # From AGVLA (base Evo1 model) stage 2 training (only LIBERO spatial) (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_libero/stage2_only_spatial/step_best" # From AGVLA (base Evo1 model) stage 2 training (only LIBERO spatial) (on A40)
### Checkpoints trained with both HabitatSim and LIBERO datasets (combined dataset) WITH DOUBLE REL STATE CONVERSION ISSUE ###
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage1/step_best" # From AGVLA (parallel action head) stage 1 training, on hm3d_1-6 and LIBERO-Spatial only (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2/step_best" # From AGVLA (parallel action head) stage 2 training, on hm3d_1-6 and LIBERO-Spatial only (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume/step_best" # From AGVLA (parallel action head) stage 2 training, on hm3d_1-6 and LIBERO-Spatial only (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage1_parallel_full_hm3d_1-2_libero_spatial/step_best" # From AGVLA (parallel action head) stage 1 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_parallel_full_hm3d_1-2_libero_spatial/step_best" # From AGVLA (parallel action head) stage 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_parallel_full_hm3d_1-2_libero_spatial_resume/step_best" # From AGVLA (parallel action head) stage 2 resume training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage1_long_full_hm3d_1-2_libero_spatial/step_best" # From AGVLA (parallel action head) stage 1 long training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage1_resume_long_full_hm3d_1-2_libero_spatial/step_best" # From AGVLA (parallel action head) stage 1 resume long training, on full hm3d_1-2 and LIBERO Spatial (on A40)
### Checkpoints trained with both HabitatSim and LIBERO datasets (combined dataset) ###
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage1_full_hm3d_1-2_libero_spatial/step_best" # From AGVLA (parallel action head) stage 1 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_full_hm3d_1-2_libero_spatial/step_best" # From AGVLA (parallel action head) stage 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage1_long_full_hm3d_1-2_libero_spatial/step_best" # From AGVLA (parallel action head) stage 1 long training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_PCGrad_full_hm3d_1-2_libero_spatial_correct_loss_split/step_best" # From AGVLA (parallel action head) PCGrad (with correct loss split) stage 1 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_PCGrad_full_hm3d_1-2_libero_spatial_dsapi/step_best" # From AGVLA (parallel action head) PCGrad (with DeepSpeed APIs) stage 1 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_PCGrad_full_hm3d_1-2_libero_spatial_dsapi/step_10000" # From AGVLA (parallel action head) PCGrad (with DeepSpeed APIs) stage 1 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_per_task_norm_split/step_best" # From AGVLA (parallel action head) PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_per_task_norm_split/step_best" # From AGVLA (parallel action head) PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume2_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_per_task_norm_split/step_best" # From AGVLA (parallel action head) PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume3_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_per_task_norm_split/step_best" # From AGVLA (parallel action head) PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume 3 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
### NEW CHECKPOINTS, with different action head architectures ###
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage1_diff_heads_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 1 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_diff_heads_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume_diff_heads_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume2_diff_heads_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
### NEW CHECKPOINT, with body-relative states and actions ###
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage1_diff_heads_body_rel_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 1 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_diff_heads_body_rel_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume_diff_heads_body_rel_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume2_diff_heads_body_rel_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume4_diff_heads_body_rel_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 4 resume 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume6_diff_heads_body_rel_PCGrad_full_hm3d_1-2_libero_spatial_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 6 resume 2 training, on full hm3d_1-2 and LIBERO Spatial (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume_diff_heads_body_rel_PCGrad_full_hm3d_1-6_libero_1-3_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 6 resume 2 training, on full hm3d_1-6 and LIBERO Spatial, Object and Goal (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume6_diff_heads_body_rel_PCGrad_full_hm3d_1-6_libero_1-3_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 6 resume 6 training, on full hm3d_1-6 and LIBERO Spatial, Object and Goal (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume10_diff_heads_body_rel_PCGrad_full_hm3d_1-6_libero_1-3_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume 10 training, on full hm3d_1-6 and LIBERO Spatial, Object and Goal (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume13_diff_heads_body_rel_PCGrad_full_hm3d_1-6_libero_1-3_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume 13 training, on full hm3d_1-6 and LIBERO Spatial, Object and Goal (on A40)
CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_combined/stage2_resume20_diff_heads_body_rel_PCGrad_full_hm3d_1-6_libero_1-3_dsapi_task_split/step_best" # From AGVLA (parallel action head), diff action head depth, body relative states and actions, PCGrad (with DeepSpeed APIs, per task normalisation loss splits) stage 2 resume 20 training, on full hm3d_1-6 and LIBERO Spatial, Object and Goal (on A40)

### Function for loading model and normaliser ###
# Specifically for inference
def load_model_and_normalizer(ckpt_dir, robot_key):
    config = json.load(open(os.path.join(ckpt_dir, "config.json"))) # Load model config
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json"))) # Load normalisation stats (used by Normaliser)

    # Override some model configs for model INFERENCE
    config["finetune_vlm"] = False # Disable VLM finetuning
    config["finetune_action_head"] = False # Disable action head finetuning
    config["num_inference_timesteps"] = 32 # Set action head flow-matching steps for action generation

    model = AGVLA(config).eval() # Create model architecture, and set to inference mode
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt") # Define checkpoint path

    checkpoint = torch.load(ckpt_path, map_location="cpu") # Load checkpoint
    model.load_state_dict(checkpoint["module"], strict=True) # Load weights into model
    model = model.to("cuda") # Move model to device

    normalizer = Normalizer(stats, robot_key) # Use normilsation stats loaded earlier, for preprocessing and postprocessing
    return model, normalizer


### Function for formatting input image for a vision model ###
# Converts an image as a Python LIST into a PyTorch TENSOR
# i.e. Image is a nested list of pixel vals
def decode_image_from_list(img_list):
    img_array = np.array(img_list, dtype=np.uint8) # Convert list to numpy array, stored as 0-255 ints
    img = cv2.resize(img_array, (448, 448)) # Resize image to 448x448 (output image shape will be 448x448x3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert colour channels to RGB (for use with PIL)
    pil = Image.fromarray(img) # Create a PIL image object
    return transforms.ToTensor()(pil).to("cuda") # Convert PIL image to PyTorch tensor, rearranging dims and normalising, and moves tensor to device


### Function for INFERENCE from a client request ###
# Client request is a JSON dict
# NOTE: Default embodiment_ids=None, as if no embodiment_ids input, then BOTH FlowmatchingActionHead and ParallelActionHead has logic implementation that defaults dataset input type to LIBERO, and acts runs logic accordingly
# So, embodiment_ids for LIBERO is None, and embodiment_ids for HabitatSim is torch.tensor([1], device="cuda") (which is passed in as the input)
def infer_from_json_dict(data: dict, model, normalizer, embodiment_ids=None):
    # Determine device and model type
    device = "cuda"
    model_dtype = next(model.parameters()).dtype

    # Decode images (from list to tensors)
    images = [decode_image_from_list(img) for img in data["image"]]
    assert len(images) == 3, "Must provide exactly 3 images." # Verify number of camera images (must be 3)
    for img in images: # Verify each image size (3x448x448)
        ### ADDED for DEBUGGING
        print(">> agvla_server.py IMG DEBUGGING <<\n", flush=True)
        print(f"imges length: {len(images)}\n", flush=True)
        print(f"img shape: {img.shape}\n", flush=True)
        ###
        assert img.shape == (3, 448, 448), "image_size must be (3,448,448)"

    # Process robot state
    state = torch.tensor(data["state"], dtype=torch.float32, device=device) # Convert incoming state to tensor
    if state.ndim == 1: # Add batch dim
        state = state.unsqueeze(0)
    if state.shape[1] < 24: # Pad states to 24 dims with zeros (duplicate of that done in Normaliser)
        state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32) # Normalise state (from raw state to [-1, 1]), and force state input to float32

    # Extract prompt (language instruction)
    prompt = data["prompt"]
    
    # Build masks
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([data["action_mask"]],dtype=torch.int32, device=device)

    # DEBUG mask prints
    print(f"image_mask,{image_mask}")
    print(f"action_mask,{action_mask}")
    
    # Run policy
    with torch.no_grad() and torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16): # Disable gradient tracking and use mixed precision
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask, ### ADDED for ParallelActionHead implementation
            embodiment_ids = embodiment_ids ### ADDED for ParallelActionHead implementation
        )
        action = action.reshape(1, -1, 24) # Reshape output to batch x time horizon x action dim
        print(f"Normalised actions: {action[0]}", flush=True) # DEBUGGING
        action = normalizer.denormalize_action(action[0]) # Denormalise actions. action[0] removes batch dim, which is not used in denormalisation
        return action.cpu().numpy().tolist() # Convert to JSON-friendly format


### Function for real-time server communication loop ###
# Receives observations from client, runs model, then sends back predicted actions
async def handle_request(websocket, model, normalizer):
    print("Client connected")
    try:
        async for message in websocket:
           
            json_data = json.loads(message) # Receive message and parse JSON into a dict
            print(f"Received JSON observation")
            # actions = infer_from_json_dict(json_data, model, normalizer) # Run inference (likely output shape: [T, 24]), FOR single action head implementation
            actions = infer_from_json_dict(json_data, model, normalizer, embodiment_ids=torch.tensor([LIBERO_EMBODIMENT_ID], dtype=torch.long, device="cuda")) # Run inference (likely output shape: [T, 24]), FOR parallel action head implementation
            await websocket.send(json.dumps(actions)) # Send result back to client
            print("Sent action chunk")

    # If client disconnects
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

### Main function ###
if __name__ == "__main__": # Start server only if this file is explicitly run (not imported as a module)
    print("Loading EVO_1 model...")
    model, normalizer = load_model_and_normalizer(CKPT_DIR, "libero_franka")

    async def main():
        print(f"EVO_1 server running at ws://0.0.0.0:{PORT}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer),
            "0.0.0.0", PORT, max_size=100_000_000
        ):
            await asyncio.Future()

    asyncio.run(main())