# These function definitions convert:
# HabitatSim outputs --> Evo1-format inputs
# Evo1-format ouputs --> HabitatSim-format outputs
# According to this code structure (from model_runner.py):
###############################################################################
# # Build model input
# example = {
#     "observation/image": img_array, # Current observation
#     "observation/ref_image": self.ref_image_array, # Start observation
#     "observation/state": state, # Current state
#     "task": self.instruction # Natural language instruction
# }

# # Run model inference
# output_all = infer(policy, example)
# output = output_all[9] # Get last predicted action
# new_coords = output[:4].tolist() # Extract first four outputs as coordinates
###############################################################################

### Import packages ###
import numpy as np
import cv2

### Script definitions ###
_START_STATE = None # Starting state for a new episode/trajectory, for trajectory-relative state logic

### Function for resetting starting state for a new episode/traj, for traj-relative state logic
def reset_start_state():
    global _START_STATE # Configure this variable to refer to the module-level/global variable (not a new local variable)
    _START_STATE = None # Reset the starting state

### Function for computing trajectory-relative states
def compute_rel_states(original_state):
    global _START_STATE # Configure this variable to refer to the module-level/global variable (not a new local variable)

    original_state = np.asarray(original_state, dtype=np.float32) # Convert input state to a np array
    print(f"Before relative state computation: {original_state}", flush=True) # DEBUGGING

    # If the starting state is not yet set (which means current step is the start of a new episode/traj evaluation)
    if _START_STATE is None:
        _START_STATE = original_state.copy() # Set the starting state to current state (COPY to prevent accidentally modifying original variable)

    rel_state = original_state.copy() # Make a copy of current original state, to prevent accidentally modifying the original variable
    rel_state -= _START_STATE # Compute trajectory-relative states, i.e. Relative state = Current state - Starting state

    # Wrap yaw angles
    rel_state[3] = (rel_state[3] + np.pi) % (2*np.pi) - np.pi

    return rel_state

### Function for converting HabitatSim STATES to Evo1-format STATES ###
def process_hs_to_evo1_states(hs_state):
    #####################
    # Transform HabitatSim states into Evo1 state input format
    # FORMATS:
    #   HabitatSim output state format: np array, len = 4, float32
    #   Evo1 input state format: np array, len = anything smaller or equal to 24, float32-compatible (i.e. dtype that is able to safely be converted into float32)
    #
    # HabitatSim inputs are: [x, y, z, yaw], where
    #   x: x position in world frame, in meters
    #   y: y position in world frame, in meters
    #   z: z height in meters
    #   yaw: yaw in degrees, global heading (wraps around 360 degrees)
    #
    # Evo1-LIBERO states are: [x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper1, gripper2]
    #   x: End-effector x pos
    #   y: End-effector y pos
    #   z: End-effector z pos
    #   axis_angle1: First component of end-effector orientation (rad)
    #   axis_angle2: Second component of end-effector orientation (rad)
    #   axis_angle3: Third component of end-effector orientation (rad)
    #   gripper1: Position/state of left finger
    #   gripper2: Position/state of right finger
    #####################
    # Convert absolute states to trajectory-relative states
    hs_state = compute_rel_states(hs_state) # NOTE: COMMENT FOR HABITATSIM DATASET CONVERSION TO LEROBOT FORMAT, BUT UNCOMMENT FOR INFERENCE
    
    # Extract HabitatSim states
    hs_x, hs_y, hs_z, hs_yaw = hs_state # NOTE: this unpacking works whether input is a list, tuple, or numpy array (with EXACTLY 4 elements)

    # Convert HabitatSim yaw values from deg to rad (which Evo1-format expects)
    # hs_yaw = np.deg2rad(hs_yaw) # DELETE

    # Build Evo1-format states
    evo1_states = np.array([hs_x, hs_y, hs_z, 0.0, 0.0, hs_yaw, 0.0, 0.0], dtype=np.float32)

    return evo1_states


### Function for converting HabitatSim IMAGES to Evo1-format IMAGES ###
def process_hs_to_evo1_imgs(hs_img):
    ####################
    # Transform HabitatSim images into Evo1 image input format (i.e. a list of images, each a numpy array of uint8)
    # FORMATS:
    #   HabitatSim output image format: np array, shape = (H, W, 3), uint8, RGB
    #   Evo1 input image format: np array, shape = (H, W, 3), uint8, BGR
    # NOTE: No need to convert each image to a list first (reducing unecessary overhead), as decode_image_from_list() converts them back to numpy arrays, and images from HabitatSim are already numpy arrays.
    ####################
    # Convert HabitatSim images from RGB to BGR (for feeding into Evo1 pipeline, i.e. into decode_image_from_list())
    # hs_img = hs_img[:, ::-1, :]
    # hs_img = np.flipud(hs_img)
    evo1_img = cv2.cvtColor(hs_img, cv2.COLOR_RGB2BGR)
    # This is needed I think. Originally, LIBERO images (DIRECTLY FROM SIMULATOR) seem to be BGR, so decode_image_from_list() (in agvla_server.py) converts BGR2RGB for inference
    #   Directly from sim is important, as training script does not seem to do any BGR2RGB conversions, meaning training data images are saved as RGB, and model is trained on RGB images
    # However, for HabitatSim eval, get_img() from test_sim.py (which is called by sim_runner.py) does RGB to BGR conversion, probably because HabiatSim images are actually RGB here, but IndoorUAV model or HabitatSim pipeline needed BGR
    # In our case, because we are using AGVLA model, which was trained on RGB images, conversion of BGR to RGB is what is needed.

    return evo1_img

### Function for converting HabitatSim outputs to Evo1-format inputs ###
def habitatsim_out_to_evo1_in(habitatsim_out):
    # Extract HabitatSim inputs
    curr_obs_img = habitatsim_out["observation/image"] # Current observation image
    curr_ref_img = habitatsim_out["observation/ref_image"] # Current reference image
    curr_state = habitatsim_out["observation/state"] # Current state
    task = habitatsim_out["task"] # Current text prompt

    # Convert HabitatSim images to Evo1-format images
    curr_obs_img = process_hs_to_evo1_imgs(curr_obs_img)
    curr_ref_img = process_hs_to_evo1_imgs(curr_ref_img)

    # Construct the images list for outputting into Evo1 pipeline
    images = [
        curr_obs_img,
        curr_ref_img,
        curr_obs_img
    ]

    # Transform HabitatSim states into Evo1 state input format
    #state = curr_state
    # curr_state[3] = np.deg2rad(curr_state[3]) # Convert yaw angle from degree to radians first, as HabitatSim output angles are in deg, but model expects rad angles
    state = process_hs_to_evo1_states(curr_state)

    ####################
    # Transform HabitatSim task into Evo1 input prompt format
    # FORMAT:
    #   HabitatSim output task format: String
    #   Evo1 input prompt format: String
    ####################
    prompt = task

    # Define the image_mask and action_mask
    image_mask = [1, 1, 0] # Only use first two images
    action_mask = [1] * 7 + [0] * 17 # Only use first 7 action entries

    # Construct Evo1-format input dictionary
    evo1_in = {
        "image": images,
        "image_mask": image_mask,
        "prompt": prompt,
        "state": state,
        "action_mask": action_mask
    }

    return evo1_in


### Function for converting Evo1-format outputs to HabitatSim inputs ###
def evo1_out_to_habitatsim_in(evo1_out, habitatsim_out):
    ####################
    # Evo1-format model outputs are: [delta_x, delta_y, delta_z, delta_axis_angle1, delta_axis_angle2, delta_axis_angle3, delta_gripper]
    # HabitatSim simulator inputs are: [x, y, z, yaw] coordinates
    # So must add these position deltas (pos changes) to the current HabitatSim state/coordinates, and feed that as 'action' into HabitatSim
    # i.e. Convert Evo1 7D action output to HabitatSim 4D coordinate input
    # Discard the Evo1-format roll, pitch and gripper action outputs (especially since in HabitatSim, drone/'agent' is always flat, so no roll or pitch changes ever)
    ####################
    # Get the last action from Evo1 output horizon
    # NOTE that these actions are already denormalised from infer_from_json_dict(), so these outputs should be real-world deltas (not normalised deltas), and can be directly added to HabitatSim states
    final_evo1_out = evo1_out[0]#[-1] # NOTE: VARIABLE NAME INCORRECT, SHOULD BE first_action, BUT NOT CHANGED YET FOR DEBUGGING
    # final_evo1_out = np.sum(evo1_out[:5], axis=0) # TEST: Using a 50-step rollout sum for getting the final HabitatSim coordinate
    # TODO: Check if final_evo1_out (evo1 output) is a list of numpy arrays, and if this type of indexing is allowed
    # evo1_out = np.asarray(evo1_out)
    # final_evo1_out = np.array([
    #     np.sum(evo1_out[:4, 0]), # x
    #     np.sum(evo1_out[:4, 1]), # y
    #     np.sum(evo1_out[:4, 2]), # z
    #     np.sum(evo1_out[:2, 3]), # roll
    #     np.sum(evo1_out[:2, 4]), # pitch
    #     np.sum(evo1_out[:2, 5]), # yaw
    #     np.sum(evo1_out[:2, 6]) # gripper
    # ])

    # Obtain the current HabitatSim state coordinates
    curr_state = habitatsim_out["observation/state"] # Current state
    hs_x, hs_y, hs_z, hs_yaw = curr_state # Extract HabitatSim states
    print(f"yaw before conversion: {hs_yaw}", flush=True)
    # hs_yaw = np.deg2rad(hs_yaw) # Convert HabitatSim yaw angles from deg to rad (for computation), as HabitatSim angle output are in deg, but model is trained, and outputs angles in rads

    # Obtain the new HabitatSim state coordinate changes
    evo1_dx, evo1_dy, evo1_dz = final_evo1_out[0:3] # x, y, z deltas
    evo1_dyaw = final_evo1_out[5] # Yaw delta
    # evo1_dyaw = np.rad2deg(evo1_dyaw) # Convert Evo1-format delta yaw from rad to deg (as HabitatSim uses angles in deg)
    # print(f"adapters output deltas: [{evo1_dx:.4f}, {evo1_dy:.4f}, {evo1_dz:.4f}, {evo1_dyaw:.4f}]", flush=True) # DEBUGGING
    # print(f"adapters output deltas: [{evo1_dx:.4f}, {evo1_dy:.4f}, {evo1_dz:.4f}, {final_evo1_out[3]:.4f}, {final_evo1_out[4]:.4f}, {evo1_dyaw:.4f}, {final_evo1_out[6]:.4f}]", flush=True) # DEBUGGING
    # evo1_dz = 0 # NOTE: DEBUGGING, comment out for real use

    print(f"hs_yaw: {hs_yaw}", flush=True)
    print(f"evo1_dyaw: {evo1_dyaw}", flush=True)

    # Construct HabitatSim-format input dictionary, calcualte the new HabitatSim state coordinates, as action inputs into HabitatSim
    # Since HabitatSim pipeline obtains last output from a 10 horizon length action output, insert 9 dummy action outputs before the actual output
    # TODO: Check if HabitatSim input is supposed to be a list of np arrays, with type float32
    dummy_zeros = np.zeros((9, 4), dtype=np.float32) # Make dummy actions to fill the first 9 actions in the predicted horizon (which are not used anyways)
    
    # Construct HabitatSim action input
    habitatsim_in = np.array([[
        (hs_x + evo1_dx), # The first output is the new x coordinate
        (hs_y + evo1_dy), # The second output is the new y coordinate
        (hs_z + evo1_dz), # The third output is the new z coordinate
        ((hs_yaw + evo1_dyaw) % (2*np.pi))# 360) # The fifth output is the new yaw angle (%360 for dealing with yaw wrap-around, i.e. if angle exceeds 360 deg, do angle%360=angle_new)
        # NOTE that HabitatSim expects angle inputs in deg, so convert from rad to deg
    ]], dtype=np.float32) # Make this shape (1,4)

    habitatsim_in = np.vstack((dummy_zeros, habitatsim_in)) # Append the dummy actions to the actual used action

    return habitatsim_in