####################
# This script converts HabitatSim datasets (used in IndoorUAV) to LeRobot v2.1 dataset format, specifically for aligning for usage with the LIBERO evaluator
# OVERVIEW STEPS:
#   Each vla_ins segment for each trajectory is a self-contained task segment
#   Iterate over all episodes (each vla_in segment in each traj_* is one episode)
#   For each vla_in segment in each traj_X file, build a time-aligned sequence (i.e. aligned with each timestep index t) of:
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
#
#   Each vla_ins segment will become one LeRobot episode
####################

### Import packages ###
import os
import sys
from datetime import datetime
import shutil
import json
import numpy as np
import argparse # For HPC job arguments parsing
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
# DATASET_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV_ALL_extracted" # Dataset directory
# VLA_INS_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV_ALL_extracted/vla_ins" # vla_ins directory
# CONVERTED_DATASET_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV_lerobot" # Converted dataset directory
# DEBUG_LOG_PATH = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/debug/dataset_converter.log" # DEBUG text file directory

# FOR single trajectory testing (i.e. converting a single trajectory)
DATASET_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV_single_traj" # Dataset directory
VLA_INS_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV_single_traj/vla_ins" # vla_ins directory
CONVERTED_DATASET_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV_single_traj_lerobot" # Converted dataset directory
DEBUG_LOG_PATH = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/debug/dataset_converter_single_hm3d_9.log" # DEBUG text file directory

#########################
### Class definitions ###
#########################
### TrajectorySegment container class definition ###
# Container class for storing data of a segment (a vla_ins segment) for a trajectory
# TODO: Check if the variable type definitions are correct here!!!
@dataclass
class TrajectorySegment:
    scene_group: str # Scene group, e.g. "hm3d_14"
    scene_id: str # Scene ID, e.g. "1Rg1SS1dRpG"
    traj_id: str # Trajectory ID, e.g. "traj_9"
    seg_id: str # Segment ID i.e. vla_ins_* ID, e.g. "vla_ins_3"

    instruction: str
    images: list[Path]
    ref_images: list[Path]
    states: list[np.ndarray]
    actions: list[np.ndarray]

### Trajectory container class definition ###
# Container class for storing all vla_ins segments for a trajectory
@dataclass
class Trajectory:
    scene_group: str # Scene group, e.g. "hm3d_14"
    scene_id: str # Scene ID, e.g. "1Rg1SS1dRpG"
    traj_id: str # Trajectory ID, e.g. "traj_9"
    segments: List[TrajectorySegment]

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

    # Return each image Path object (images here is a list of Path objects)
    return images


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


### Helper function to format LIBERO states for computing LIBERO-format delta actions ###
# Formats LIBERO states for calculating LIBERO-format actions, i.e. [delta_x, delta_y, delta_z, delta_angle1, delta_angle2, delta_angle3, delta_gripper]
# So this function changes the 8D states ([x, y, z, angle1, angle2, angle3, gripper1, gripper2]) to [x, y, z, angle1, angle2, angle3, gripper] format
#   gripper1 and gripper2 should always be the same magnitude and opposite in sign, ALTHOUGH during navigation in HabitatSim, they are both actually always 0, so just merge into a single 0
#   angle1, angle2 and gripper are always 0, as in HabitatSim agent does not pitch/roll or have a gripper
# Angles here are already in rad, so no need to convert
def format_libero_states(states):
    s = np.asarray(states, dtype=np.float32) # Convert hs_states for easier indexing
    return np.array([s[0], s[1], s[2], s[3], s[4], s[5], 0.0], dtype=np.float32) # Format and return states


### Compute actions corresponding to current trajectory segment (a vla_ins) ###
# Function converts states of trajectory segment into actions taken, via: action[t] = state[t+1] - state[t]
# This conversion changes ABSOLUTE TRAJECTORY into RELATIVE MOTION
# Inputs: States throughout current trajectory segment - Input is a list of state vectors, state = [s0, s1, s2, ..., sT], where each state is something like [x, y, z, angle1, angle2, angle3, gripper1, gripper2]
#   NOTE States are NOT [x, y, z, yaw] as the states fed into this function have already been converted to LIBERO 8D states (and angles from deg to rad) in load_states().
# Outputs: Actions taken between each trajectory segment timestep - Is a list of numpy arrays, [(s1-s0), (s2-s1), ..., 0]
# NOTE that outputs are also converted to LIBERO outputs format, i.e. [delta_x, delta_y, delta_z, delta_angle1, delta_angle2, delta_angle3, delta_gripper]
# ALTHOUGH delta_angle1, delta_angle2, and delta_gripper will always be 0 as HabitatSim agent is always flat against horizontal (no pitch/roll), and it doesn't use gripper during navigation
# ALSO HabitatSim angles are in degrees, but LIBERO uses radian angles
def compute_delta_actions(states_raw):
    actions = [] # Initialise empty list to store computed delta actions for current trajectory segment

    # Format LIBERO states for calculating delta actions
    states = [format_libero_states(s) for s in states_raw]

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


### Function for loading a trajectory's segments (the vla_ins corresponding to that trajectory) ###
# Loads all VLA instruction files corresponding to a trajectory, and returns them as a list of segments
# Each segment contains:
#   Instruction agent should follow
#   Frames of trajectory the instruction applies to
# Structure of a vla_ins_*.json file is:
#   {
#       "instruction": "Text instruction",
#       "source": [start_frame, end_frame]
#   }
# INPUTS: Directory for a single trajectory
# OUTPUTS: All instruction segments for the input trajectory, as a list ([]) of dictionaries ({}), where each dictionary contains key-value pairs for instruction, start and end frame for each vla_ins traj segment
def load_vla_segments(vla_ins_dir):
    # Create empty list to hold info about all vla_ins segments in traj
    segments = []

    # Loop through all vla_ins instruction files, sorted
    for vla_file in sorted(Path(vla_ins_dir).glob("vla_ins_*.json")):
        print(f"Reading vla_ins: {vla_file}", flush=True) # DEBUGGING

        # NOTE: Some files contain corrupted, non-UTF-8 characters, which cause json.load to fail. Skip these files
        try:
            with open(vla_file, "r") as f: # Read one vla_ins instruction file
                data = json.load(f) # Read JSON contents

            # Extract relevant fields of JSON file, and append to segments
            segments.append({
                "instruction": data["instruction"],
                "start_frame": data["source"][0],
                "end_frame": data["source"][1],
                "file_name": vla_file.stem, # E.g. "vla_ins_1"
            })
        # IF file threw an error
        except Exception as e:
            print(f"    SKIPPED: {vla_file} | {type(e).__name__}: {e}", flush=True) # Print out error for skipping the file
            with open(DEBUG_LOG_PATH, "a") as f: f.write(f"SKIPPED vla_ins: {vla_file} | {type(e).__name__}: {e}\n") # Write to DEBUG file
            continue
    
    return segments

    
####################
# This function reads one trajectory file (traj_*), and returns the:
#   instruction,
#   images,
#   states,
#   actions,
# associated with each vla_in segment corresponding to that trajectory file
####################
# INPUTS: Trajectory directory, vla_ins_*.json directory, and the trajectory's scene group, scene ID, and trajectory ID
# OUTPUTS: List[TrajectorySegment] - Trajectory's information (instruction, images, states, actions), and its scene group, scene ID, and trajectory ID, in a dataclass
def load_trajectory(traj_dir, vla_ins_dir, scene_group, scene_id, traj_id):
    traj_dir = Path(traj_dir) # Convert the input trajectory directory to a PATH object
    
    # Load information corresponding to current trajectory
    #instruction = load_instruction(traj_dir) # Load INSTRUCTION # NOTE: Moved to inside the loop, as each segment has a different instruction
    images = load_images(traj_dir / "screenshots") # Load IMAGES
    states = load_states(traj_dir / "posture.json") # Load STATES
    #actions = load_actions(traj_dir / "real_action.json") # Load ACTIONS
    #actions = compute_delta_actions(states) # Compute ACTIONS # NOTE: Moved to inside the loop, computing delta actions after slicing, so final zero action is at end of EACH TASK, not only at end of original traj (making every LeRobot episode consistent)
    segments = load_vla_segments(vla_ins_dir) # Load VLA_INS SEGMENTS of current traj

    trajectory_segs = [] # Initialise empty list to store trajectory segments (vla_ins info for each traj)

    # Loop through all segments (vla_ins) in the trajectory
    for segment in segments:
        # Get start and end frame indices of the current segment
        start_frame = segment["start_frame"] - 1
        end_frame = segment["end_frame"]

        # Extract the start and end frames
        seg_states = states[start_frame:end_frame]
        seg_imgs = images[start_frame:end_frame]
        seg_ref_imgs = [images[start_frame]] * (end_frame - start_frame) # The reference image for all timesteps in each segment is just the starting frame
        seg_actions = compute_delta_actions(seg_states) # Compute delta actions here, to ensure all segments end with a zero action (instead of just at the end of the original traj)

        # Sanity check for whether frame indexing is correct
        assert len(seg_imgs) == (end_frame - start_frame), (
            f"Segment image indexing incorrect for {traj_dir}, file{segment['file_name']}: "
            f"start_frame: {start_frame}, "
            f"end_frame: {end_frame}, "
            f"image length = {len(seg_imgs)}"
        )
        
        # Sanity check for checking number of timesteps match for data
        # Number of timesteps for images, states, and actions must match
        assert len(seg_imgs) == len(seg_ref_imgs) == len(seg_states) == len(seg_actions), (
            f"Length mismatch in {traj_dir}, file {segment['file_name']}: "
            f"{len(seg_imgs)} images, "
            f"{len(seg_ref_imgs)} ref images, "
            f"{len(seg_states)} states, "
            f"{len(seg_actions)} actions"
        )

        # Append current trajectory segment to all trajectory segment data
        trajectory_segs.append(
            TrajectorySegment(
                scene_group = scene_group,
                scene_id = scene_id,
                traj_id = traj_id,
                seg_id = segment["file_name"],

                instruction=segment["instruction"],
                images = seg_imgs,
                ref_images = seg_ref_imgs,
                states = seg_states,
                actions = seg_actions
            )
        )
    
    # Return trajectory data
    return trajectory_segs


### Load scene group dataset of multiple trajectories ###
# This function builds a list of episodes by loading each trajectory information in a scene group dataset folder (e.g. hm3d)
# Trajectory information means all data in all segments (vla_ins) in a trajectory
# Inputs: Scene group dataset directory, vla_ins folder directory (e.g. if scene group is hm3d_14/, then vla_ins_dir is vla_ins/hm3d_14)
# Outputs: List of episodes in the scene group dataset directory, each containing that episode's instruction, images, states, and actions (for writing the LeRobot dataset)
def load_scene_group(scene_group_dir, scene_group_vla_ins_dir):
    scenes_data = [] # Initialise empty list to store all loaded trajectories (all trajectories in each scene in a scene group)
    scene_group_dir = Path(scene_group_dir) # Convert scene group directory to a Path object
    scene_group_vla_ins_dir = Path(scene_group_vla_ins_dir) # Convert scene group vla_ins directory to a Path object
    scene_group = scene_group_dir.name # Get the scene group ID of scene group being processed for scenes (e.g. "hm3d_14")

    # Loop over all scenes in the scene group (sorted)
    for scene_dir in sorted(scene_group_dir.iterdir()):
        # Skip all non-directories (anything not a folder, e.g. README files, as only folders are scenes)
        if not scene_dir.is_dir():
            continue
        
        scene_id = scene_dir.name # Get the current scene ID (e.g. "1Rg1SS1dRpG")
        trajectories_data = [] # Initialise empty list to store all trajectory information in current scene

        # Loop over all trajectories in each scene (trajectories all start with "traj_"). Each traj is a single trajectory folder
        for traj_dir in sorted(scene_dir.glob("traj_*")):
            traj_id = traj_dir.name # Get current trajectory ID (e.g. "traj_-1")

            # NOT ALL TRAJECTORIES HAVE CORRESPONDING VLA_INS SEGMENTS!!
            # Check if current trajectory exists as a directory in vla_ins folder for corresponding scene
            curr_vla_ins_dir = scene_group_vla_ins_dir / scene_id / traj_id # Construct corresponding vla_ins directory for current trajectory (e.g. vla_ins/hm3d_14/1Rg1SS1dRpG/traj_-1)
            if not curr_vla_ins_dir.is_dir(): # Skip trajectories that have no corresponding vla_ins directory
                print(f"Skipping {scene_group}/{scene_id}/{traj_id}: No corresponding vla_ins directory", flush=True)
                with open(DEBUG_LOG_PATH, "a") as f: f.write(f"SKIPPED traj file: {scene_group}/{scene_id}/{traj_id} | No corresponding vla_ins directory\n") # Write to DEBUG file
                continue

            # Load current trajectory information (i.e. all segments in traj)
            traj_segs = load_trajectory(
                traj_dir,
                curr_vla_ins_dir, # The vla_ins folder directory for current trajectory
                scene_group = scene_group,
                scene_id = scene_id,
                traj_id = traj_id
            )

            # Place current trajectory information into a Trajectory dataclass
            trajectory = Trajectory(
                scene_group = scene_group,
                scene_id = scene_id,
                traj_id = traj_id,
                segments = traj_segs
            )
            
            trajectories_data.append(trajectory) # Add current trajectory segments information to the list of all trajectory segments in current scene
            # NOTE: trajectories_data here is a list of Trajectory dataclass objects
        
        scenes_data.append( # Add current scene data (i.e. data all trajectory segments in current scene) to list of all scene data
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
            for segment in traj.segments:
                # Collect all states and actions of all trajectories
                # extend() adds each element of each entry individually (e.g. each x state of every trajectory is added up)
                # Resulting states and actions will still be same dim as a single entry
                states.extend(segment.states)
                actions.extend(segment.actions)
    
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

### HPC command-line input argument parser function ###
# This function extracts the input arguments fed into the function via an HPC job (e.g. --scene_group)
# OUTPUTS: Command-line argument provided
def parse_args():
    # Create an ArgumentParser object, for defining arguments the program accepts
    parser = argparse.ArgumentParser(description="Convert HabitatSim dataset to LeRobot format")

    # Add expected command-line argument
    parser.add_argument("--scene_group", type=str, required=True, help="Scene group to convert, e.g. hm3d_1")

    # Actually read command line, and return parsed argument attributes
    return parser.parse_args()

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
        repo_id = f"IndoorUAV-VLA_lerobot/{output_dir.name}", # Dataset's name on HuggingFace Hub

        root = output_dir, # Define where to write converted dataset files to (where episodes, videos, metadata, etc. will go)

        fps = 20, # Frame rate (sampling rate for each trajectory) in frames per second. Important for syncing video and actions

        use_videos = True, # Flag to store images as videos (instead of sequence of images), which is more efficient

        robot_type = "indooruav", # Robot type specification, just used for metadata only

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
                        "gripper", "gripper"
                    ]
                },
            },

            "action": { # Agent action (at each timestep)
                "dtype": "float32",
                "shape": (7,), # 7D action vector
                "names": {
                    "motors": [
                        "x", "y", "z",
                        "axis_angle1", "axis_angle2", "axis_angle3",
                        "gripper"
                    ]
                }
            },

            # "task": { # Task description (per episode)
            #     "dtype": "string",
            #     "shape": (1,),
            # },

            # Required v2.1 bookkeeping fields
            "timestamp": {"dtype": "float32", "shape": (1,)},
            "frame_index": {"dtype": "int64", "shape": (1,)},
            "episode_index": {"dtype": "int64", "shape": (1,)},
        },
    )

    return dataset


### Function for writing one trajectory segment file ###
# Writes one trajectory segment (one vla_ins segment) as one LeRobot episode, writing it into the LeRobot dataset frame by frame
#   Function iterates through every timestep of a trajectory segment, loads each image, pairs it with corrsponding state, action, and task instruction,
#   adds each timestep (frame) to current LeRobot episode, and then saves completed episode to the dataset.
# INPUTS:
#   dataset - A LeRobotDataset object
#   trajectory_seg - A TrajectorySegment object containing: images, states, actions, instruction
# OUTPUTS: Writes one complete LeRobot episode
def write_trajectory_seg(dataset, trajectory_seg):
    # Loop through every trajectory segment timestep (zip combined all four lists together, where each iter processes the images, state, and action from the same timestep). Guarantees synchronisation
    # The inputs lists (images, ref_images, states, actions) MUST have same length, otherwise zip will stop at the shortest list
    for image_path, ref_image_path, state, action in zip(
        trajectory_seg.images,
        trajectory_seg.ref_images,
        trajectory_seg.states,
        trajectory_seg.actions,
    ):
        
        # Load image and reference image through input path, convert to RGB, then convert to numpy (matching dataset schema defined in adapters and generally LeRobot image format)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        ref_image = np.asarray(Image.open(ref_image_path).convert("RGB"))

        # Add one frame (append one timestep to current episode)
        # Dictionary contains every observation at that timestep
        dataset.add_frame(
            {
                "observation.images.image": image,
                "observation.images.ref_image": ref_image,
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                # "task": trajectory_seg.instruction, # Every frame gets the same instruction (as every frame needs to know what task agent is performing) # NOTE: FOR LeRobot 0.4.4
            },
            task=trajectory_seg.instruction, # NOTE: For LeRobot 0.3.3
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
        print(f"Processing scene {scene.scene_id}", flush=True)

        # Loop over every trajectory in the current scene
        for trajectory in scene.trajectories:
            # Print current trajectory ID
            print(f"    Writing {trajectory.traj_id}", flush=True)

            # Loop over every segment in the current trajectory
            for trajectory_seg in trajectory.segments:
                # Write the current trajectory (every trajectory becomes one episode)
                write_trajectory_seg(
                    dataset,
                    trajectory_seg,
                )

                # Increment episode count
                num_episodes += 1
        
    # Finalise dataset (after every trajectory in every scene has been written)
    #   Writes metadata files, build indexes, finalises video files, computes dataset stats, and makes dataset ready for loading and training
    #   May cause dataset to be incomplete or miss metadata if not called
    # dataset.consolidate()

    # Print final number of written episodes
    print(f"Wrote {num_episodes} episodes.", flush=True)


#####################
### MAIN FUNCTION ###
#####################
# def main(): # UNCOMMENT if want running converter from HPC/terminal with NO input args
### UNCOMMENT if running converter from HPC with specific scene group to convert input args
def main(scene_group_to_convert):
    global DEBUG_LOG_PATH # Ensure the module/script-level DEBUG_LOG_PATH variable is being modified (not a new local variable)
    DEBUG_LOG_PATH = Path(DEBUG_LOG_PATH) # Convert DEBUG_LOG_PATH to Path object
    debug_log_path_parent = DEBUG_LOG_PATH.parent # Obtain parent directory of DEBUG_LOG_PATH (i.e. the directory without last directory section)
    DEBUG_LOG_PATH = debug_log_path_parent / f"data_converter_{scene_group_to_convert}.log" # Construct new DEBUG_LOG_PATH specific for current scene group
###

    # Initialisations
    with open(DEBUG_LOG_PATH, "w") as f: f.write("") # Clear debug text file
    with open(DEBUG_LOG_PATH, "a") as f: f.write(f"Run: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n") # Write to DEBUG file to indicate script running
    with open(DEBUG_LOG_PATH, "a") as f: f.write(f"Debug log file directory: {DEBUG_LOG_PATH}\n") # Write to DEBUG file the debug log file directory

    # Loop over every scene group
    for scene_group_dir in sorted(Path(DATASET_DIR).iterdir()):
        # Skip all non-directories (as all datasets are in folders/directories)
        if not scene_group_dir.is_dir():
            continue

        # Get current scene group
        scene_group = scene_group_dir.name

        # Skip all folders that are not actual datasets (i.e. the vla_ins/, scene_datasets/, and without_screenshot/ folders)
        if scene_group in {
            "vla_ins",
            "scene_datasets",
            "without_screenshot",
        }:
            continue

        # DEBUGGING: For just converting a single scene group, manually setting the scene group to convert
        # if not scene_group in {
        #     "hm3d_1",
        # }:
        #     continue

        # Only process the requested scene group (from HPC job input args)
        # COMMENT OUT if not running HPC job with input args
        if scene_group != scene_group_to_convert: # If current scene group is not the one requested from HPC job input args
            continue

        # Construct vla_ins/ directory for current scene group (vla_ins/ directory has same directory structure as datasets, but with vla_ins/ before the actual scene group folder)
        vla_ins_dir = Path(VLA_INS_DIR) / scene_group

        # Skip scene groups that do NOT have corresponding VLA instructions
        if not vla_ins_dir.is_dir():
            print(f"Skipping {scene_group}: No corresponding vla_ins folder")
            with open(DEBUG_LOG_PATH, "a") as f: f.write(f"Skipped {scene_group}: No corresponding vla_ins folder\n") # Write to DEBUG file the skipped scene group
            continue

        # Construct output directory
        converted_output_dir = Path(CONVERTED_DATASET_DIR) / scene_group
        
        ### FOR DEBUGGING: Clear converted dataset destination directory for clean conversion run (as LeRobot dataset creation expects a completely fresh directory)
        # Comment out at the end though, to prevent accidental dataset rewriting (i.e. requires manual deletion of existing dataset for this script to work)
        # if os.path.exists(converted_output_dir):
        #     print(f"Clearing converted dataset destination directory: {converted_output_dir}", flush=True)
        #     shutil.rmtree(converted_output_dir)
        ###

        ### Start data conversion ###
        print(f"\n>> Processing {scene_group} <<\n")
    
        # Load all trajectory data from all scenes of scene group
        with open(DEBUG_LOG_PATH, "a") as f: f.write(f"[LOADING] SCENE GROUP DATA: {scene_group}\n") # Write to DEBUG file for indicating start of scene group data loading
        scenes = load_scene_group(scene_group_dir, vla_ins_dir)

        # # DEBUGGING: Take only first scene, and only first trajectory in scene
        # scenes = scenes[:1]
        # for scene in scenes:
        #     scene.trajectories = scene.trajectories[:1]

        #     for traj in scene.trajectories:
        #         traj.segments = traj.segments[:1]

        # Convert all loaded scene data into LeRobot format, and write to the output directory
        # Iterates over all trajectories in all scenes, and writes each trajectory as a LeRobot episode
        with open(DEBUG_LOG_PATH, "a") as f: f.write(f"[WRITING] SCENE GROUP DATA: {scene_group}\n") # Write to DEBUG file for indicating writing scene group data
        write_scene_group(
            scenes,
            output_dir = converted_output_dir,
        )
        with open(DEBUG_LOG_PATH, "a") as f: f.write(f"[WRITING] FINISHED SCENE GROUP DATA: {scene_group}\n") # Write to DEBUG file for indicating finished writing scene group data

### Prevent code execution if loaded as a module ###
if __name__ == "__main__":
    # main() # UNCOMMENT if running converter from HPC/terminal with NO input args
    
    # UNCOMMENT if running converter from HPC WITH input args
    args = parse_args()
    main(args.scene_group)