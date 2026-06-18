### Import packages ###
import json
import torch

### Normaliser class definition ###
class Normalizer:
    def __init__(self, stats_or_path):

        # Loading normalisation statistics
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32) # Convert input values to tensor
            if x.shape[0] < 24: # Pad inputs
                pad = torch.zeros(24 - x.shape[0], dtype=torch.float32)
                x = torch.cat([x, pad], dim=0)
            elif x.shape[0] > 24:
                raise ValueError(f"Input length {x.shape[0]} exceeds expected 24")
            return x

        if len(stats) != 1: # Verify only one robot exists
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")

        # Extract robot statistics
        robot_key = list(stats.keys())[0]
        robot_stats = stats[robot_key]

        # Load robot state and action statistics
        self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        self.action_min = pad_to_24(robot_stats["action"]["min"])
        self.action_max = pad_to_24(robot_stats["action"]["max"])

    ### Function for normalising state ###
    # State is converted into the range: [-1, 1]
    # Min-max normalisation used: 2*(state - state_min)/(state_max - state_min) - 1
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        # Move stats to device
        state_min = self.state_min.to(state.device, dtype=state.dtype)
        state_max = self.state_max.to(state.device, dtype=state.dtype)
        return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)

    ### Function for denormalising action ###
    # Action is converted back into actual action units and range
    # Via: (action + 1)/2 --> Maps [-1, 1] to [0, 1]
    # (action_max - action_min) scales it
    # + action_min shifts that back into original action range
    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1) # Add a batch dim
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min