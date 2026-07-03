####################
# This script converts HabitatSim datasets (used in IndoorUAV) to LeRobot v2.1 dataset format, specifically for aligning for usage with the LIBERO evaluator
# OVERVIEW STEPS:
#   Iterate over all episodes (each traj_* is one episode)
#   For each traj_X file, build a time-aligned sequence (i.e. aligned with each timestep index t) of:
#       RGB image
#       State / pose
#       Action
#       Language instruction
#   Conversion of states and actions must be done to match Evo1-format and HabitatSim inputs and outputs
#   --> These form per-timestep dataset rows, i.e. LeRobot tabular dataset (Parquet)
#
#   Create video files (required by LeRobot v2.1)
#       For each episode, convert screenshots to MP4 (used by LeRobot for fast image loading)
#
#   Generate epsiode metadata
#       Including: epsiodes.jsonl, tasks.jsonl, info.json
#
#   Compute normalisation statistics (IMPORTANT)
#       Across entire dataset, calculate:
#           state stats: mean(state), std(state)
#           actions stats: mean(action), std(action)
#       Save these into episodes_stats.jsonl
####################

### Import packages ###
import os
import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

### Import modules ###
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Obtain the project root
sys.path.append(PROJECT_ROOT) # Append project root to current path (allowing agvla_server.py to be discoverable)
print("PROJECT_ROOT: ", PROJECT_ROOT, flush=True) # DEBUGGING: Print project root to check
import IndoorUAV_eval.adapters as adapters

### Script definitions ###
DATASET_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV/hm3d_14/1Rg1SS1dRpG/traj_9" # Dataset directory
CONVERTED_DATASET_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV/converted/hm3d_14/1Rg1SS1dRpG/traj_9" # Converted dataset directory

#########################
### Class definitions ###
#########################
### Trajectory container class definition ###
# TODO: Check if the variable type definitions are correct here!!!
@dataclass
class Trajectory:
    scene_group: str # Scene group, e.g. "hm3d_14"
    scene_id: str # Scene ID, e.g. "1Rg1SS1dRpG"
    traj_id: str # Trajectory ID, e.g. "traj_9"

    instruction: str
    images: list[Path]
    states: list[np.ndarray]
    actions: list[np.ndarray]

### Scene dataclass definition ###
# This dataclass stores the trajectory information in each scene, and their corresponding scene IDs for use
@dataclass
class Scene:
    scene_group: str # Scene group, e.g. "hm3d_14"
    scene_id : str # Scene ID, e.g. "1Rg1SS1dRpG"
    trajectories: List[Trajectory] # NOTE: Trajectory must be defined before this class, otherwise use forward referencing: "Trajectory"


########################
### Helper functions ###
########################
### This function loads a JSON file, and returns it ###
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    

### Load images file paths (sorted) corresponding to current trajectory ###
# Inputs: Folder path of images
# Outputs: A sorted list of image file paths (as strings), e.g. ["data/0.png", "data/1.png", "data/2.png", ...]
# Images are in the screenshots/ folder, labelled like: 1.png, 2.png, ..., 10.png, 11.png, ...
def load_images(image_dir):
    images = sorted( # Sort filenames in numeric order (1, 2, 10), NOT alphabetic order ("1", "10", "2")
        Path(image_dir).glob("*.png"), # Find all PNG files in image directory (files ending in .png), returning an iterator of file paths (e.g. img_0.png, img_1.png, img_2.png, ...)
        key=lambda x: int(x.stem) # Get filename without .png extension, and convert to int
    )

    # Return each image Path object as a string path
    return [str(i) for i in images]


### Load instruction corresponding to current trajectory ###
# Inputs: Current trajectory directory
# Outputs: The instruction string
# instruction.json has format: {"instruction": "Instruction string..."}
def load_instruction(traj_dir):
    instruction_path = traj_dir / "instruction.json" # Construct path for current instruction.json
    return load_json(instruction_path)["instruction"] # Return instruction string


### Load states (posture) corresponding to the current trajectory ###
# Inputs: Current posture file (posture.json) directory
# Outputs: A list of numpy array for the states (each of shape (4,)) - i.e. A time-ordered list of states: [array([x,y,z,yaw]), array([...]), ...]
# The data format in posture.json is: [x, y, z, yaw] per frame
def load_states(posture_path):
    # Load JSON file
    # Data will look like a list of frames (lists), each frame a list of 4 numbers: [[x,y,z,yaw], [x,y,z,yaw], ...]
    with open(posture_path, "r") as f:
        data = json.load(f)
    
    # Convert each frame into numpy arrays, of float32 (using list comprehension)
    states = [
        adapters.process_hs_to_evo1_states( np.asarray(frame, dtype=np.float32) ) # Apply the HabitatSim to Evo1-format image state transformation function before storing (changes states from 4D to 8D)
        for frame in data
    ]

    return states


### Compute actions corresponding to current trajectory ###
# Function converts state trajectory into actions taken, via: action[t] = state[t+1] - state[t]
# This conversion changes ABSOLUTE TRAJECTORY into RELATIVE MOTION
# Inputs: States throughout current trajectory - Input is a list of state vectors, state = [s0, s1, s2, ..., sT], where each state is something like [x, y, z, yaw]
# Outputs: Actions taken between each trajectory timestep - Is a list of numpy arrays, [(s1-s0), (s2-s1), ..., 0]
def compute_delta_actions(states):
    actions = [] # Initialise empty list to store computed delta actions for current trajectory

    # Loop over all timesteps/frames EXCEPT last frame (as each step needs a next state to compute delta actions)
    for t in range(len(states) - 1):
        delta = states[t+1] - states[t] # Calculate movement in x, y, z, and changes in yaw (like a velocity signal). Here, two numpy arrays are subtracted
        actions.append(delta) # Store calculated delta action to total list of actions
    
    actions.append(np.zeros_like(states[0])) # Last frame has no next state, so assign a zero action (as can't compute a real delta)

    return actions

# OLD action logic that directly reads in actions from real_action.json
# ACTION_ORDER = [
#     "fly_forward",
#     "fly_backward",
#     "fly_left",
#     "fly_right",
#     "fly_up",
#     "fly_down",
#     "turn_left",
#     "turn_right",
#     "stop",
# ]

# ACTION_INDEX = {a: i for i, a in enumerate(ACTION_ORDER)}

# def load_actions(action_path):
#     """
#     Converts real_action.json → fixed vector per frame
#     Shape: (num_frames, 9)
#     """

#     with open(action_path, "r") as f:
#         data = json.load(f)

#     actions = []

#     for frame_entry in data["frame"]:

#         vec = np.zeros(len(ACTION_ORDER), dtype=np.float32)

#         for name, value in frame_entry["actions"].items():

#             if name not in ACTION_INDEX:
#                 continue

#             vec[ACTION_INDEX[name]] = value

#         actions.append(vec)

#     return actions

    
####################
# This function reads one trajectory file (traj_*), and returns the:
#   instruction,
#   images,
#   states,
#   actions,
# associated with that trajectory file
####################
# Inputs: Trajectory directory, and the trajectory's scene group, scene ID, and trajectory ID
# Outputs: Trajectory's information (instruction, images, states, actions), and its scene group, scene ID, and trajectory ID, in a dataclass
def load_trajectory(traj_dir, scene_group, scene_id, traj_id):
    traj_dir = Path(traj_dir) # Convert the input trajectory directory to a PATH object
    
    # Load information corresponding to current trajectory
    instruction = load_instruction(traj_dir) # Load INSTRUCTION
    images = load_images(traj_dir / "screenshots") # Load IMAGES
    states = load_states(traj_dir / "posture.json") # Load STATES
    #actions = load_actions(traj_dir / "real_action.json") # Load ACTIONS
    actions = compute_delta_actions(states) # Compute ACTIONS

    # Sanity check for checking number of timesteps match for data
    # Number of timesteps for images, states, and actions must match
    assert len(images) == len(states) == len(actions), (
        f"Length mismatch in {traj_dir}: "
        f"{len(images)} images, "
        f"{len(states)} states, "
        f"{len(actions)} actions"
    )

    # Return trajectory data
    return Trajectory(
        scene_group = scene_group,
        scene_id = scene_id,
        traj_id = traj_id,

        instruction=instruction,
        images = images,
        states = states,
        actions = actions
    )


### Load scene group dataset of multiple trajectories ###
# This function builds a list of episodes by loading each trajectory information in a scene group dataset folder (e.g. hm3d)
# Inputs: Scene group dataset directory
# Outputs: List of episodes in the scene group dataset directory, each containing that episode's instruction, images, states, and actions (for writing the LeRobot dataset)
def load_scene_group(scene_group_dir):
    scenes_data = [] # Initialise empty list to store all loaded trajectories (all trajectories in each scene in a scene group)
    scene_group_dir = Path(scene_group_dir) # Convert scene group directory to a Path object
    scene_group = scene_group_dir.name # Get the scene group ID of scene group being processed for scenes (e.g. "hm3d_14")

    # Loop over all scenes in the scene group (sorted)
    for scene_dir in sorted(scene_group_dir.iterdir()):
        # Skip all non-directories (anything not a folder, e.g. README files, as only folders are scenes)
        if not scene_dir.is_dir():
            continue
        
        scene_id = scene_dir.name # Get the current scene ID
        trajectories_data = [] # Initialise ampty list to store all trajectory information in current scene

        # Loop over all trajectories in each scene (trajectories all start with "traj_"). Each traj is a single trajectory folder
        for traj_dir in sorted(scene_dir.glob("traj_*")):
            traj_id = traj_dir.name # Get current trajectory ID
            traj_data = load_trajectory( # Load current trajectory information
                traj_dir,
                scene_group = scene_group,
                scene_id = scene_id,
                traj_id = traj_id
            )
            
            trajectories_data.append(traj_data) # Add current trajectory information to the list of all trajectories in current scene
        
        scenes_data.append( # Add current scene data (i.e. data all trajectories in current scene) to list of all scene data
            Scene(
                scene_group = scene_group,
                scene_id = scene_id,
                trajectories = trajectories_data
            )
        )

    # Return all scenes data in current scene group
    return scenes_data


### Function for computing dataset normalisation statistics (mean, standard deviation) for all states and actions in a dataset ###
# Computes global norm stats (not per-trajectory stats), by aggregating all states and actions from all trajectories, and computes per-dimension mean and std
# Used for pre- and post-processing of data
# INPUTS: All trajectory data of all scenes in a scene group
# OUTPUTS: A dictionary of the dataset normalisation statistics
def compute_dataset_stats(scene_group_data):
    # Create empty states and actions lists to store all states and actions from entire dataset
    states = []
    actions = []

    # Loop over all trajectories in all scenes in scene group
    for scene in scene_group_data:
        for traj in scene.trajectories:
            # Collect all states and actions of all trajectories
            # extend() adds each element of each entry individually (e.g. each x state of every trajectory is added up)
            # Resulting states and actions will still be same dim as a single entry
            states.extend(traj.states)
            actions.extend(traj.actions)
    
    # Convert to numpy arrays (giving shape: total timesteps x state/action dim)
    states = np.asarray(states)
    actions = np.asarray(actions)

    # Compute normalisation stats
    # Compute statistics column-/feature-wise (axis=0), to calc states per state dim
    stats = {
        "state_mean": states.mean(axis=0),
        "state_std": states.std(axis=0),
        "action_mean": actions.mean(axis=0),
        "action_std": actions.std(axis=0),
    }

    return stats

####################
# LeRobot writer functions
####################
### Function for creating an EMPTY LeRobot dataset ###
# Defines what dataset will contain, but doesn't add any data yet
# INPUTS: Directory were converted dataset will be stored locally
# OUTPUTS: A structured dataset
def create_lerobot_dataset(output_dir):
    # Create dataset object
    dataset = LeRobotDataset.create(
        #repo_id = "IndoorUAV/hm3d_evo1", # Dataset's name on HuggingFace Hub

        root = output_dir, # Define where to write converted dataset files to (where episodes, videos, metadata, etc. will go)

        fps = 10, # Frame rate (sampling rate for each trajectory) in frames per second. Important for syncing video and actions

        use_videos = True, # Flag to store images as videos (instead of sequence of images), which is more efficient

        # Feature schema - define structure of each timestep
        features = {
            "observation.images.image": { # RGB observations - image
                "dtype": "video", # Store as video frames (each timestep contains an image)
                "shape": (720, 1280, 3), # Height x Width x Channels (RGB)
                "names": ["height", "width", "rgb"],
            },

            "observation.images.ref_image": { # RGB observations - image
                "dtype": "video", # Store as video frames (each timestep contains an image)
                "shape": (720, 1280, 3), # Height x Width x Channels (RGB)
                "names": ["height", "width", "rgb"],
            },

            "observation.state": { # Robot state
                "dtype": "float32",
                "shape": (8,), # 8D state vector
                "names": {
                    "motors": [
                        "x", "y", "z",
                        "axis_angle1", "axis_angle2", "axis_angle3",
                        "gripper1", "gripper2"
                    ]
                },
            },

            "action": { # Agent action (at each timestep)
                "dtype": "float32",
                "shape": (7,),
            },

            "task.language_instruction": { # Task description (per episode)
                "dtype": "string",
            },

            # Required v2.1 bookkeeping fields
            "timestamp": {"dtype": "float32", "shape": (1,)},
            "frame_index": {"dtype": "int64", "shape": (1,)},
            "episode_index": {"dtype": "int64", "shape": (1,)},
        },
    )

    return dataset


### Function for writing one trajectory file ###
# Writes one trajectory as one LeRobot episode, writing it into the LeRobot dataset frame by frame
#   Function iterates through every timestep of a trajectory, loads each image, pairs it with corrsponding state, action, and task instruction,
#   adds each timestep (frame) to current LeRobot episode, and then saves completed episode to the dataset.
# INPUTS:
#   dataset - A LeRobotDataset object
#   trajectory - A trajectory object containing: images, states, actions, instruction
# OUTPUTS: Writes one complete LeRobot episode
def write_trajectory(dataset, trajectory):
    # Loop through every trajectory timestep (zip combined all three lists together, where each iter processes the image, state, and action from the same timestep). Guarantees synchronisation
    # The inputs lists (images, states, actions) MUST have same length, otherwise zip will stop at the shortest list
    for image_path, state, action in zip(
        trajectory.images,
        trajectory.states,
        trajectory.actions,
    ):
        
        # Load image through input path, convert to RGB, then convert to numpy (matching dataset schema defined in adapters and generally LeRobot image format)
        image = np.asarray(Image.open(image_path).convert("RGB"))

        # Add one frame (append one timestep to current episode)
        # Dictionary contains every observation at that timestep
        dataset.add_frame(
            {
                "observation.image": image,
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                "task": trajectory.instruction, # Every frame gets the same instruction (as every frame needs to know what task agent is performing)
            }
        )

    # Save episode (after all frames/timesteps from trajectory are added)
    # This command finalises metadata, writes video (MP4), stores states/actions, increments episode counter, and prepares for the next episode (without this, LeRobot will assume still adding frames to current episode)
    dataset.save_episode()


### Function for writing whole dataset (i.e. whole scene group) ###
# Function creates a LeRobot dataset, writes every trajectory from every scene in a scene group as a separate episode, then finalises dataset
# INPUTS:
#   scene_group_data - A collection of Scene objects (from the input scene group)
#   output_dir - Directory to save the converted LeRobot dataset
# OUTPUTS: Writes LeRobot dataset for all trajectories in all scenes in a scene group
def write_scene_group(scene_group_data, output_dir):
    # Create an empty LeRobot dataset (with schema observation.image, observation.state, action, task)
    dataset = create_lerobot_dataset(output_dir)

    # Create counter for number of episodes written
    num_episodes = 0

    # Loop over every scene in current scene group
    for scene in scene_group_data:
        # Print which scene is being processed
        print(f"Processing scene {scene.scene_id}")

        # Loop over every trajectory in the current scene
        for trajectory in scene.trajectories:
            # Print current trajectory ID
            print(f"    Writing {trajectory.traj_id}")

            # Write the current trajectory (every trajectory becomes one episode)
            write_trajectory(
                dataset,
                trajectory,
            )

            # Increment episode count
            num_episodes += 1
        
    # Finalise dataset (after every trajectory in every scene has been written)
    #   Writes metadata files, build indexes, finalises video files, computes dataset stats, and makes dataset ready for loading and training
    #   May cause dataset to be incomplete or miss metadata if not called
    dataset.consolidate()

    # Print final number of written episodes
    print(f"Wrote {num_episodes} episodes.")


#####################
### MAIN FUNCTION ###
#####################
# def main():
#     # Load all trajectory data from all scenes of scene group
#     scenes = load_scene_group(DATASET_DIR)

#     # Convert all loaded scene data into LeRobot format, and write to the output directory
#     # Iterates over all trajectories in all scenes, and writes each trajectory as a LeRobot episode
#     write_scene_group(
#         scenes,
#         output_dir = CONVERTED_DATASET_DIR,
#     )

# ### Prevent code execution if loaded as a module ###
# if __name__ == "__main__":
#     main()