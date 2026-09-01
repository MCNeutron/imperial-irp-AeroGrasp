##########
# This script takes in the full test_val.json file for all vla_ins testing trajectories, and extracts only the ones specified
##########
### Import packages ###
import json
import re

### Script definitions ###
# File directories
INPUT_FILE = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/IndoorUAV_eval/test_vla_hm3d_all.json"
OUTPUT_FILE = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/IndoorUAV_eval/test_vla_hm3d_7-10_easy_100.json"
# OUTPUT_FILE = "/rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/IndoorUAV_eval/test_vla_hm3d_1-6.json"

# Scene groups you want to keep
SCENE_GROUPS = {
    # "hm3d_1",
    # "hm3d_2",
    # "hm3d_3",
    # "hm3d_4",
    # "hm3d_5",
    # "hm3d_6",
    "hm3d_7",
    "hm3d_8",
    "hm3d_9",
    "hm3d_10",
}

# Difficulties you want to keep
DIFFICULTIES = {
    "easy",
    # "medium",
    # "hard",
}

# Action types you want to keep
#
# Set to None to keep ALL action types.
#
# Examples:
# ACTION_TYPES = None
# ACTION_TYPES = {"forward"}
# ACTION_TYPES = {"turn", "forward"}
# ACTION_TYPES = {"turn"}
ACTION_TYPES = None

# Define the number of trajectories to evaluate
NUM_TRAJS = 100

### Filtering ###
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

filtered_data = {}

for path, info in data.items():

    # Stop once desired number of trajectories to evaluate is reached
    if len(filtered_data) >= NUM_TRAJS:
        break

    # Extract scene group from path, e.g.
    # /hm3d_14/1Rg1SS1dRpG/traj_-3/vla_ins_1.json
    match = re.search(r"/(hm3d_\d+)/", path)

    if match is None:
        continue

    scene_group = match.group(1)

    # Check scene group
    scene_match = scene_group in SCENE_GROUPS

    # Check difficulty
    difficulty_match = info["difficulty"] in DIFFICULTIES

    # Check action type
    if ACTION_TYPES is None:
        # None means keep all action types
        action_type_match = True
    else:
        entry_action_types = set(info.get("action_type", []))

        # Keep if ALL specified action types are present
        action_type_match = ACTION_TYPES.issubset(entry_action_types)

    # Keep entry only if all conditions are satisfied
    if scene_match and difficulty_match and action_type_match:
        filtered_data[path] = info

### Save extracted entries ###
with open(OUTPUT_FILE, "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"Original entries: {len(data)}")
print(f"Filtered entries: {len(filtered_data)}")
print(f"Removed entries: {len(data) - len(filtered_data)}")
print(f"Saved to: {OUTPUT_FILE}")