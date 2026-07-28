####################
# This script takes in a single task instruction, then outputs the corresponding HabitatSim LeRobot ground truth actions
# This is for validation of whether the converted dataset and the HabitatSim evaluation pipeline (when using the Evo1 model
# architecture) actually works. If the pipeline can't replicate the actual ground truth actions (even without using model
# inference), then something is wrong with evaluation pipeline.
####################

### Import packages ###
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

### Class definition ###
# This class replays ground truth actions from a LeRobot parquet episode
#   It searches task.jsonl files for the task_index of the current instruction
#   Then it extracts actions from searching through parquet files for the actions with matching task_index
class GroundTruthPolicy:
    ##########
    # Class initialisation
    # INPUTS:
    #   dataset_dir (str): LeRobot scene group directory (e.g. IndoorUAV_lerobot/hmd_8) - This directory is where the actions will be searched for
    #   task (str): Natural language task instruction
    # OUTPUTS: None
    ##########
    def __init__(self, dataset_dir):
        # Save dataset path
        self.dataset_dir = Path(dataset_dir)
        print(f"Current dataset path: {self.dataset_dir}", flush=True)

        # Replay counter
        self.current_step = 0

        # Initialise variables that persist across multiple calls
        self.task_index = None
        self.actions = []
        self.prev_task = None
    
    ### Function for finding task index ###
    # This function goes through meta/tasks.jsonl and finds the corresponding task_index for the current input task string
    # INPUTS:
    #   task (str): The current task string
    # OUTPUTS:
    #   task_index: The corresponding task_index for the current task
    def find_task_index(self, task):
        # Construct the tasks.jsonl directory path
        tasks_file_path = self.dataset_dir / "meta" / "tasks.jsonl" # E.g. IndoorUAV_lerobot/hm3d_8/meta/tasks.jsonl

        # Read the tasks.jsonl file, and find the corresponding task_index for current input task string
        with open(tasks_file_path, "r") as f:
            # Loop through all lines in the tasks.jsonl file
            for line in f:
                entry = json.loads(line)

                # Find the corresponding task_index for current task
                if entry["task"] == task: # Check if current line matches the current query task
                    return entry["task_index"]
        
        # If run to this point, it means corresponding task_index was not found
        print(f"Task not found...", flush=True)
        raise ValueError(f"Task not found:\n{task}")
    
    ### Function for loading actions corresponding to current task ###
    # This function searches all parquet files in the data directory, and extracts actions matching the task_index corresponding to the current task
    # INPUTS: None
    # OUTPUTS:
    #   The actions found corresponding to the current task
    def load_actions(self, task_index):
        # Initialise an empty list for storing the corresponding actions
        actions = []

        # Construct the data directory 
        data_dir = self.dataset_dir / "data"

        ### Search all parquet files for the actions corresponding to the current task_index ###
        # Obtain parquet files (sorted)
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        print(f">> Searching {len(parquet_files)} parquet files <<", flush=True) # Print this for visualisation

        # Loop through all parquet files found
        for parquet_file in parquet_files:
            # Load only required columns
            df = pd.read_parquet(
                parquet_file,
                columns=["task_index", "action"]
            )

            # Get rows that have the same task_index as current task
            matched_rows = df[ df["task_index"] == task_index ]

            # If FOUND rows with task_index as current task
            if len(matched_rows) > 0:
                print(f"Found {len(matched_rows)} frames in {parquet_file.name}", flush=True) # Print results
                actions.extend(matched_rows["action"].tolist()) # Append the rows with task_index matching current task to actions list
        
        # If found no rows with task_index matching current task
        if len(actions) == 0:
            raise RuntimeError("No actions found for task")
        
        return actions
    
    ### Function for resetting the current step counter ###
    # INPUTS: None
    # OUTPUTS: None
    def reset(self):
        self.current_step = 0 # Reset replay trajectory

    ### Function for getting the next ground truth action ###
    # Returns the next ground truth action from the extracted list of actions for current task
    # INPUTS:
    #   The task
    # OUTPUTS:
    #   The next (ground truth) action to send to HabitatSim
    def get_action(self, task):
        # Check if current task is a new query or not. If new, then find and extract its corresponding actions from parquet files
        if self.prev_task != task:
            print(f">> New task query detected <<")

            # Find task index corresponding to the current task
            self.task_index = self.find_task_index(task)
            print(f"Task index found: {self.task_index}", flush=True)

            # Obtain actions corresponding to current task
            self.actions = self.load_actions(self.task_index)
            print(f"Loaded {len(self.actions)} actions", flush=True)

            # Reset current steps counter
            self.reset()

            # Record current task
            self.prev_task = task

        # Check if all the extracted actions have been executed
        if self.current_step >= len(self.actions):
            print("Ground truth trajectory execution complete")
            return np.zeros(7, dtype=np.float32)
        
        # (Otherwise) extract the current action to execute, based on current step counter
        action = self.actions[self.current_step] # One step
        print(f"GT output deltas: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, {action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}, {action[6]:.4f}]", flush=True) # DEBUGGING
        # if self.current_step <= len(self.actions): # Two steps
        #     action = self.actions[self.current_step] + self.actions[self.current_step+1] # Two steps
        # if self.current_step+2 <= len(self.actions): # Three steps
        #     action = self.actions[self.current_step] + self.actions[self.current_step+1] + self.actions[self.current_step+2] # Three steps
        # else:
        #     action = self.actions[self.current_step]
        # print(f"GT output deltas after sum: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, {action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}, {action[6]:.4f}]", flush=True) # DEBUGGING
        dummy_zeros = np.zeros((9,7), dtype=np.float32) # Create dummy zeros for appending to action, to create a 'fake' horizon prediction
        action = np.vstack((action[None,:], dummy_zeros)) # Append dummy zeros to action prediction

        self.current_step += 1 # Increment current step counter after action execution

        # Return action for sending to HabitatSim
        return np.asarray(action, dtype=np.float32)