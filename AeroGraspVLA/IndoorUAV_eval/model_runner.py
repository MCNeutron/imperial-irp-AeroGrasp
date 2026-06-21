### Import packages ###
import os
import json
import time
import numpy as np
from PIL import Image

### Import modules ###
import agvla_server as agvla

### Script definitions ###
# Define model checkpoint directory
CKPT_DIR = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/weights/agvla_libero_evo1_weights" ### ADDED - For LIBERO


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
def infer(policy, inputs):
    # TODO: Adapter for converting HabitatSim inputs to Evo1 inputs
    
    evo1_format_actions = agvla.infer_from_json_dict(evo1_format_inputs, policy, normalizer) # Run inference (likely output shape: [T, 24])

    # TODO: Adapter for converting Evo1 action outputs to HabitatSim actions
    # First just make a dummy output adapter, just to see if things run

    #return policy.infer(inputs)["actions"]
    return actions

### Load model immediately ###
# Import at module level
#policy = init_model()
policy, normalizer = agvla.load_model_and_normalizer(CKPT_DIR)

### Define shared-folder architecture ###
# Shared folder holds simulator inputs and outputs, for communication between model and simulator
SHARED_FOLDER = "shared_folder"
MODEL_INPUT_DIR = os.path.join(SHARED_FOLDER, "model_input") # Observations from controller received here
MODEL_OUTPUT_DIR = os.path.join(SHARED_FOLDER, "model_output") # Model predictions written here
INSTRUCTIONS_DIR = os.path.join(SHARED_FOLDER, "instructions") # Instructions for current episode stored here
os.makedirs(MODEL_INPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)

### Definition for class that manages inference ###
class ModelService:
    def __init__(self):
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
            output_all = infer(policy, example)
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
    print("模型推理服务启动...")
    model_service = ModelService()

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
