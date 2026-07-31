### Import packages ###
import os
import sys
import json
import time
import numpy as np
import torch
from PIL import Image

### Import modules ###
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Obtain the project root
sys.path.append(PROJECT_ROOT) # Append project root to current path (allowing agvla_server.py to be discoverable)
print("PROJECT_ROOT: ", PROJECT_ROOT, flush=True) # DEBUGGING: Print project root to check
import agvla_server as agvla
import adapters

### Script definitions ###
HABITATSIM_EMBODIMENT_ID = 1 # Integer ID for HabitatSim embodiment
# Define model checkpoint directory
# CKPT_DIR = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/weights/agvla_libero_evo1_weights" ### ADDED - For LIBERO
# CKPT_DIR = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage1/step_best" # From AGVLA (base Evo1 model) stage1 training
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage2/step_best" # From AGVLA (base Evo1 model) stage2 training
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage2_resume/step_best" # From AGVLA (base Evo1 model) stage2 resume1 training
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/only_hm3d_8_stage1/step_best" # From AGVLA (base Evo1 model) only hm3d_8 stage1 training
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/only_hm3d_2_stage1/step_best" # From AGVLA (base Evo1 model) only hm3d_2 stage1 training
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/only_hm3d_3_stage1/step_best" # From AGVLA (base Evo1 model) only hm3d_3 stage 1 training
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/only_hm3d_9_stage1/step_best" # From AGVLA (base Evo1 model) only hm3d_9 stage 1 training
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/only_hm3d_6_stage1/step_best" # From AGVLA (base Evo1 model) only hm3d_6 stage 1 training
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/only_hm3d_4_stage1/step_best" # From AGVLA (base Evo1 model) only hm3d_4 stage 1 training
### NEW CHECKPOINTS, trained from HabitatSim datasets with yaw wrapping logic ###
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage1/step_best" # From AGVLA (base Evo1 model) stage 1 training (with yaw wrapping logic in dataset creation)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage2_resume/step_best" # From AGVLA (base Evo1 model) stage 2 resume training (with yaw wrapping logic in dataset creation)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage1_small_dataset/step_best" # From AGVLA (base Evo1 model) stage 1 training (smaller dataset)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage2_small_dataset/step_best" # From AGVLA (base Evo1 model) stage 2 training (smaller dataset) (10000 steps only)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage2_resume_small_dataset/step_best" # From AGVLA (base Evo1 model) stage 2 resume training (smaller dataset) (36570 steps only)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage1_10_horizon_single_traj/step_best" # From AGVLA (base Evo1 model) only hm3d_4 stage 1 training, with horizon of 10 (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage2_10_horizon_single_traj/step_best" # From AGVLA (base Evo1 model) only hm3d_4 stage 2 training, with horizon of 10 (only 51260 steps) (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage2_resume2_small_dataset/step_best" # From AGVLA (base Evo1 model) stage 2 resume 2 training (smaller dataset) (62500 steps only) (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage1_10_horizon/step_best" # From AGVLA (base Evo1 model) stage 1 training, 10 horizon (smaller dataset) (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage2_10_horizon/step_best" # From AGVLA (base Evo1 model) stage 2 training, 10 horizon (smaller dataset) (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage1_10_horizon_rel_state_test_rel/step_best" # From AGVLA (base Evo1 model) stage 1 training, 10 horizon, traj-rel states (only hm3d_1) (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage1_10_horizon_masked_rel_state_test_rel/step_best" # From AGVLA (base Evo1 model) stage 1 training, 4 states, 10 horizon, traj-rel states (only hm3d_1) (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage1_10_horizon_masked_rel_state_test_rel_only_single_traj_hm3d_4/step_best" # From AGVLA (base Evo1 model) stage 1 training, 4 states, 10 horizon, traj-rel states (only hm3d_4 1 traj) (on A40)
# CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage1_50_horizon_masked_rel_state_test_rel_only_single_traj_hm3d_4/step_best" # From AGVLA (base Evo1 model) stage 1 training, 4 states, 50 horizon, traj-rel states (only hm3d_4 1 traj) (on A40)
CKPT_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/checkpoints/agvla_habitatsim/stage2_10_horizon_masked_rel_state_test_rel/step_best" # From AGVLA (base Evo1 model) stage 2 training, 4 states, 10 horizon, traj-rel states (only hm3d_1) (on A40)

# Define DEBUGGING directories
DEBUG_FOLDER_DIR = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/debug/agvla_habitat" # Define debug folder directory
os.makedirs(DEBUG_FOLDER_DIR, exist_ok=True) # Ensure the debug directory exists
RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
DEBUG_FILE_DIR = os.path.join(DEBUG_FOLDER_DIR, f"adapter_debug_{RUN_TIMESTAMP}.txt") # Define debug file directory
print(f"model_runner.py debug log: {DEBUG_FILE_DIR}", flush=True) # Print debug file directory for ease of identification

### Ground truth policy debugging ###
# Define the ground truth policy to use for debugging whether the HabitatSim evaluation pipeline can recreate its training data trajectory (FROM NO MODEL INFERENCE)
# from helpers.ground_truth_policy import GroundTruthPolicy
# DATASET_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV_single_traj_lerobot_rel/hm3d_4"
# # DATASET_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV_rel_pos_test_lerobot_rel/hm3d_1"
# ground_truth_policy = GroundTruthPolicy(DATASET_DIR) # Create the ground truth policy model
###


### Function for initialising the trained model/policy ###
# def init_model():
#     from openpi.training import config
#     from openpi.policies import policy_config

#     # Load model configurations
#     config = config.get_config("pi0_fast_libero_low_mem_finetune")

#     # Load model checkpoint
#     checkpoint_dir = "/rds/general/user/ll1225/home/imperial_irp/openpi_test/openpi/checkpoints/pi0_fast_libero_low_mem_finetune/train_test/29999" # ADDED

#     # Create trained policy
#     return policy_config.create_trained_policy(config, checkpoint_dir)

### Function for inference wrapper ###
#def infer(policy, habitat_format_out):
def infer(policy, normalizer, habitat_format_out): # ADDED
    # Convert HabitatSim outputs to Evo1-format inputs
    print("BEFORE input adapter...", flush=True)
    evo1_format_in = adapters.habitatsim_out_to_evo1_in(habitat_format_out)
    print(f'Input adapter outputs (states): {evo1_format_in["state"]}', flush=True) # DEBUGGING
    print("AFTER input adapter...", flush=True)

    # Run inference (likely output shape: [T, 24])
    print("BEFORE evo1 infer...", flush=True)
    evo1_format_actions = agvla.infer_from_json_dict(evo1_format_in, policy, normalizer) ### IF using single FlowmatchingActionHead model architecture
    # print_out = agvla.infer_from_json_dict(evo1_format_in, policy, normalizer) ### IF using single FlowmatchingActionHead model architecture
    # evo1_format_actions = agvla.infer_from_json_dict(evo1_format_in, policy, normalizer, embodiment_ids=torch.tensor([HABITATSIM_EMBODIMENT_ID], dtype=torch.long, device="cuda")) ### IF using ParallelActionHead model architecture
    # evo1_format_actions = ground_truth_policy.get_action(habitat_format_out["task"]) # DEBUGGING, for testing ground truth policy (i.e. feed the training data actions directly into the HabitatSim eval pipeline)
    # print_out = ground_truth_policy.get_action(habitat_format_out["task"]) # DEBUGGING, for testing ground truth policy (i.e. feed the training data actions directly into the HabitatSim eval pipeline)
    # print(f"Evo1 infer outputs: {evo1_format_actions}", flush=True) # DEBUGGING
    # print(f"model preds: [{print_out[0][0]:.4}, {print_out[0][1]:.4}, {print_out[0][2]:.4}, {print_out[0][3]:.4}, {print_out[0][4]:.4}, {print_out[0][5]:.4}, {print_out[0][6]:.4}]", flush=True)
    print(f"gt preds: [{evo1_format_actions[0][0]:.4}, {evo1_format_actions[0][1]:.4}, {evo1_format_actions[0][2]:.4}, {evo1_format_actions[0][3]:.4}, {evo1_format_actions[0][4]:.4}, {evo1_format_actions[0][5]:.4}, {evo1_format_actions[0][6]:.4}]", flush=True)
    print("AFTER evo1 infer...", flush=True)

    # Convert Evo1-format action outputs to HabitatSim actions
    print("BEFORE output adapter...", flush=True)
    habitat_format_actions = adapters.evo1_out_to_habitatsim_in(evo1_format_actions, habitat_format_out) # Also pass in habitat_format_out to calculate new state coordinates as actions in HabitatSim
    print(f'Current state: {habitat_format_out["observation/state"]}', flush=True) # DEBUGGING
    print(f"Output adapter outputs: {habitat_format_actions}", flush=True) # DEBUGGING
    print("AFTER output adapter...", flush=True)

    # DEBUGGING
    ####################
    with open(DEBUG_FILE_DIR, "a") as f:
        f.write("\n" + "=" * 80 + "\n") # Separator

        # Log Evo1-format inputs
        f.write(">> EVO1-FORMAT INPUTS <<\n")
        f.write(
            f"prompt: {evo1_format_in['prompt']}\n"
            f"state: {evo1_format_in['state']}\n"
            f"image_mask: {evo1_format_in['image_mask']}\n"
            f"action_mask: {evo1_format_in['action_mask']}\n"
        )
        for i, img in enumerate(evo1_format_in["image"]):
            f.write(
                f"image[{i}]: "
                f"type={type(img)}, "
                f"shape={img.shape}, "
                f"dtype={img.dtype}\n"
            )
        f.write("\n")

        f.write(
            ">> EVO1-FORMAT ACTIONS <<\n"
            f"type={type(evo1_format_actions)}\n"
            f"shape={np.asarray(evo1_format_actions).shape}\n"
            f"value=\n{evo1_format_actions}\n\n"
        )

        f.write(
            f">> HABITATSIM ACTIONS <<\n"
            f"type={type(habitat_format_actions)}\n"
            f"shape={habitat_format_actions.shape}\n"
            f"dtype={habitat_format_actions.dtype}\n"
        )
        f.write(f"value={habitat_format_actions}\n")
    ####################

    #return policy.infer(inputs)["actions"]
    return habitat_format_actions

### Load model immediately ###
# Import at module level
#policy = init_model()
#policy, normalizer = agvla.load_model_and_normalizer(CKPT_DIR) # ATTEMPT to move it to main

### Define shared-folder architecture ###
# Shared folder holds simulator inputs and outputs, for communication between model and simulator
#SHARED_FOLDER = "shared_folder"
SHARED_FOLDER = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/shared_folder" ### ADDED
MODEL_INPUT_DIR = os.path.join(SHARED_FOLDER, "model_input") # Observations from controller received here
MODEL_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "model_output") # Model predictions written here
INSTRUCTIONS_DIR = os.path.join(SHARED_FOLDER, "instructions") # Instructions for current episode stored here
os.makedirs(MODEL_INPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)

### Definition for class that manages inference ###
class ModelService:
    #def __init__(self):
    ### ADDED/CHANGED
    def __init__(self, policy, normalizer):
        # Set policy/model and normalizer (as model/policy and normalizer is now defined in main, not globally)
        self.policy = policy
        self.normalizer = normalizer
    ###
        self.current_episode = None
        self.instruction = None # Language instruction
        self.end_coords = None # Goal coordinates, for evaluation
        self.ref_image_array = None # The starting image
        self.last_start_image_path = None  # 跟踪上次加载的图像路径

    ### Function for updating episode metadata ###
    def load_instruction(self):
        """加载当前指令并更新参考图像"""
        instruction_file = os.path.join(INSTRUCTIONS_DIR, "current_instruction.json") # Read instruction file
        if os.path.exists(instruction_file):
            time.sleep(0.2)
            with open(instruction_file, 'r') as f:
                data = json.load(f)

            # Check if new episode
            if self.current_episode != data.get("episode_key"):
                self.current_episode = data.get("episode_key")
                self.instruction = data.get("instruction")
                self.end_coords = data.get("end_coords")
                self.last_start_image_path = None  # 重置图像路径

                ### ADDED for trajectory-relative states logic
                # Reset the starting state when a new episode/trajectory starts
                adapters.reset_start_state()
                ###

            # Get starting image path
            start_image_path = data.get("start_image_path")

            # 检查是否有新的起始图像路径
            if start_image_path and start_image_path != self.last_start_image_path:
                self.last_start_image_path = start_image_path

                # 加载参考图像
                if os.path.exists(start_image_path):
                    ref_img = Image.open(start_image_path).convert('RGB') # Load reference image (starting screenshot)
                    self.ref_image_array = np.asarray(ref_img, dtype=np.uint8)
                    print(f"更新参考图像: {os.path.basename(start_image_path)}")
                else:
                    print(f"警告: 参考图像不存在: {start_image_path}")
                    self.ref_image_array = None

    ### Function for the core inference pipeline ###
    def process_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Read incoming observation
            episode_key = data.get("episode_key", "")
            image_path = data.get("image_path", "")
            coordinates = data.get("coordinates", [])

            # Update current instruction
            self.load_instruction()

            # Check if episode matches current one (prevents stale messages from old episodes)
            if episode_key != self.current_episode:
                print(f"忽略文件: 不属于当前episode ({episode_key} vs {self.current_episode})")
                return False

            # 获取输入图像
            if not os.path.exists(image_path):
                print(f"图像文件不存在: {image_path}")
                return False

            # Load current image
            img = Image.open(image_path).convert('RGB')
            img_array = np.asarray(img, dtype=np.uint8) # This is the model input

            # Make sure coordinates have 4 entries, and pad if needed
            if len(coordinates) < 4:
                coordinates = coordinates + [0.0] * (4 - len(coordinates))

            state = np.array(coordinates[:4], dtype=np.float32) # Convert from list to np array of float32

            # Build model input
            example = {
                "observation/image": img_array, # Current observation
                "observation/ref_image": self.ref_image_array, # Start observation
                "observation/state": state, # Current state
                "task": self.instruction # Natural language instruction
            }

            # Run model inference
            #output_all = infer(policy, example)
            output_all = infer(self.policy, self.normalizer, example) # ADDED
            output = output_all[9] # Get last predicted action
            new_coords = output[:4].tolist() # Extract first four outputs as coordinates
            
            # Save prediction (to MODEL_OUTPUT_DIR)
            timestamp = time.time()
            output_file = os.path.join(MODEL_OUTPUT_DIR, f"model_output_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump({
                    "episode_key": self.current_episode,
                    "coordinates": new_coords
                }, f)

            print(f"推理完成 - 新坐标: {new_coords}")
            return True

        except Exception as e:
            print(f"处理文件 {file_path} 出错: {str(e)}")
            return False
        finally:
            # Cleanup - Delete processed observation file
            if os.path.exists(file_path):
                os.remove(file_path)

### Function for main loop ###
def main():
    ### ADDED
    # Load model
    policy, normalizer = agvla.load_model_and_normalizer(CKPT_DIR)
    print(">> Model loaded successfully", flush=True)
    ###
    
    print("模型推理服务启动...")
    #model_service = ModelService()
    model_service = ModelService(policy, normalizer) # ADDED

    try:
        while True:
            # Check for instruction updates
            model_service.load_instruction()

            # Scan input directory for new observations
            processed = False
            for file_name in os.listdir(MODEL_INPUT_DIR):
                if not file_name.endswith('.json'):
                    continue

                file_path = os.path.join(MODEL_INPUT_DIR, file_name)
                if model_service.process_file(file_path): # Process each observation
                    processed = True

            # 如果没有处理任何文件，等待一会儿
            if not processed:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("模型推理服务停止")


if __name__ == "__main__":

    main()
