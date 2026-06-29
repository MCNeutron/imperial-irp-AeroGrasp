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

### Function for converting HabitatSim outputs to Evo1-format inputs ###
def habitatsim_out_to_evo1_in(habitatsim_out):
    # Extract HabitatSim inputs
    curr_obs_img = habitatsim_out["observation/image"] # Current observation image
    curr_ref_img = habitatsim_out["observation/ref_image"] # Current reference image
    curr_state = habitatsim_out["observation/state"] # Current state
    task = habitatsim_out["task"] # Current text prompt

    ####################
    # Transform HabitatSim images into Evo1 image input format (i.e. a list of images, each a numpy array of uint8)
    # FORMATS:
    #   HabitatSim output image format: np array, shape = (H, W, 3), uint8, RGB
    #   Evo1 input image format: np array, shape = (H, W, 3), uint8, BGR
    # NOTE: No need to convert each image to a list first (reducing unecessary overhead), as decode_image_from_list() converts them back to numpy arrays, and images from HabitatSim are already numpy arrays.
    ####################
    # Convert HabitatSim images from RGB to BGR (for feeding into Evo1 pipeline, i.e. into decode_image_from_list())
    curr_obs_img = cv2.cvtColor(curr_obs_img, cv2.COLOR_RGB2BGR)
    curr_ref_img = cv2.cvtColor(curr_ref_img, cv2.COLOR_RGB2BGR)

    # Construct the images list for outputting into Evo1 pipeline
    images = [
        curr_obs_img,
        curr_ref_img,
        curr_obs_img
    ]

    #####################
    # Transform HabitatSim stage into Evo1 state input format
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
    #state = curr_state
    # Extract HabitatSim states
    hs_x, hs_y, hs_z, hs_yaw = curr_state

    # Convert HabitatSim yaw values from deg to rad (which Evo1-format expects)
    hs_yaw = np.deg2rad(hs_yaw)

    # Build Evo1-format states
    state = np.array([hs_x, hs_y, hs_z, 0.0, 0.0, hs_yaw, 0.0, 0.0], dtype=np.float32)

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
    final_evo1_out = evo1_out[-1]
    # TODO: Check if final_evo1_out (evo1 output) is a list of numpy arrays, and if this type of indexing is allowed

    # Obtain the current HabitatSim state coordinates
    curr_state = habitatsim_out["observation/state"] # Current state
    hs_x, hs_y, hs_z, hs_yaw = curr_state # Extract HabitatSim states

    # Obtain the new HabitatSim state coordinate changes
    evo1_dx, evo1_dy, evo1_dz = final_evo1_out[0:3] # x, y, z deltas
    evo1_dyaw = final_evo1_out[5] # Yaw delta
    evo1_dyaw = np.rad2deg(evo1_dyaw) # Convert Evo1-format delta yaw from rad to deg (as HabitatSim uses angles in deg)

    # Construct HabitatSim-format input dictionary, calcualte the new HabitatSim state coordinates, as action inputs into HabitatSim
    # Since HabitatSim pipeline obtains last output from a 10 horizon length action output, insert 9 dummy action outputs before the actual output
    # TODO: Check if HabitatSim input is supposed to be a list of np arrays, with type float32
    # TODO: Also, do I need to do something with the normaliser, to normalise data? CHECK THIS!
    dummy_zeros = np.zeros((9, 4), dtype=np.float32) # Make dummy actions to fill the first 9 actions in the predicted horizon (which are not used anyways)

    # TODO: ISSUE with Evo1-format outputs, in that they are likely NORMALISED outputs, and not directly the actual distance/angle changes. Must UNNORMALISE them first, before adding to HS states
    # i.e. REVERSE TRAINING NORMALISATION, though check if it is already done downstream? Maybe unnormalise here, add, then normalise again for unnormalising again downstream?
    
    # Construct HabitatSim action input
    habitatsim_in = np.array([[
        (hs_x + evo1_dx), # The first output is the new x coordinate
        (hs_y + evo1_dy), # The second output is the new y coordinate
        (hs_z + evo1_dz), # The third output is the new z coordinate
        ((hs_yaw + evo1_dyaw) % 360) # The fifth output is the new yaw angle (%360 for dealing with yaw wrap-around, i.e. if angle exceeds 360 deg, do angle%360=angle_new)
    ]], dtype=np.float32) # Make this shape (1,4)

    habitatsim_in = np.vstack((dummy_zeros, habitatsim_in)) # Append the dummy actions to the actual used action

    return habitatsim_in