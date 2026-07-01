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

### Import modules ###
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Obtain the project root
sys.path.append(PROJECT_ROOT) # Append project root to current path (allowing agvla_server.py to be discoverable)
print("PROJECT_ROOT: ", PROJECT_ROOT, flush=True) # DEBUGGING: Print project root to check
import IndoorUAV_eval.adapters as adapters

### Script definitions ###
DATASET_DIR = "/rds/general/user/ll1225/ephemeral/imperial_irp/extended_evo1/datasets/IndoorUAV/hm3d_14/1Rg1SS1dRpG/traj_9"

### Form data/ ###

### Form meta/ ###

### Form videos/ ###

