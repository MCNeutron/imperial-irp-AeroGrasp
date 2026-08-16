import sys
import os
import math
from torch import amp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import wandb
import swanlab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from model.AGVLA import AGVLA ### EDITED
from accelerate import Accelerator 
import logging
from datetime import datetime
import argparse
from accelerate import Accelerator, DistributedType
import json
import shutil
from torch.optim import AdamW

import warnings

### ADDED
### Script definitions ###
# Integer IDs for the different embodiments
from model.embodiment_id import LIBERO_EMBODIMENT_ID, HABITATSIM_EMBODIMENT_ID

# PCGrad function
from helpers.pcgrad import pcgrad_backward

# Set up logging of training metrics
import csv
curr_time = datetime.now().strftime("%Y%m%d_%H%M%S") # Get current time (for placing in training metrics log file name)
TRAINING_METRICS_PATH = f"/rds/general/user/ll1225/home/imperial_irp/extended_evo1/training_metrics_{curr_time}.csv"
os.makedirs(os.path.dirname(TRAINING_METRICS_PATH), exist_ok=True) # Ensure logging directory exists, and make it if it doesn't
###

accelerator = Accelerator()

def get_with_warning(config: dict, key: str, default):
    if key in config:
        return config[key]
    else:
        warnings.warn(f"'{key}' not found in config, using default: {default!r}")
        return default


def inspect_named_submodules(module_dict: dict, verbose: bool = True):

    total_all, trainable_all = 0, 0
    logging.info("\n Parameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info(f"\n Module: {module_name}")
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(f"Trainable {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
            elif verbose:
                logging.info(f"Frozen {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
        logging.info("-" * 70)
        logging.info(f"Total     : {total / 1e6:.2f}M")
        logging.info(f"Trainable : {trainable / 1e6:.2f}M")
        logging.info(f"Frozen    : {(total - trainable) / 1e6:.2f}M")
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info(f"ALL TOTAL     : {total_all / 1e6:.2f}M")
    logging.info(f"ALL TRAINABLE : {trainable_all / 1e6:.2f}M")
    logging.info(f"ALL FROZEN    : {(total_all - trainable_all) / 1e6:.2f}M")
    logging.info("=" * 70)


### ADDED for Parallel Action Head implementation
# This function pads the target actions and corresponding action masks to the maximum action horizon, when different horizon lengths are used across different embodiments
# INPUTS:
#   Actions in the current batch
#   Action masks for the actions
#   Maximum horizon length (to pad to)
# OUTPUTS:
#   List of padded actions (where any action with a horizon less than the maximum horizon will have the extra horizon entries padded with zeros)
#   List of corresponding padded action masks (NOTE that the masks here are JUST for masking out the extra padded horizons added, to prevent them contributing to training loss gradients)
# This function achieves for the action mask:
#   real timestep + valid dimension --> TRUE
#   real_timestep + masked dimension --> FALSE (i.e. any masked dimensions in the ORIGINAL mask)
#   padded timestep --> FALSE
def pad_actions(actions, action_masks, max_horizon):
    # Obtain the batch size and action dimensions
    batch_size = len(actions)
    action_dim = actions[0].shape[1]

    # Initialise torch arrays for storing the padded actions and padded action masks
    padded_actions = torch.zeros(batch_size, max_horizon, action_dim, dtype=actions[0].dtype,) # Initialise a padded action array with all entries as zero
    padded_masks = torch.zeros(batch_size, max_horizon, action_dim, dtype=torch.bool,) # Initialise a mask array with all mask elements as False (will fill in which are True ones after)

    # Loop through all individual actions and action mask pairs
    for i, (a, m) in enumerate(zip(actions, action_masks)):
        h = a.shape[0]
        padded_actions[i, :h] = a # Place the current action into the initial h horizons. If current action's horizon length is smaller than the max horizon, the remaining horizon elements will be 0
        padded_masks[i, :h] = m # Place the current action's mask into the initial h horizons. If the current action mask's horizon length is smaller than the max horizon, the remaining horizon elements (i.e. the padded timesteps) will remain as False (as initialised before)

    return padded_actions, padded_masks # Return the resulting padded actions and corresponding padded mask
###

def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    images = [item["images"] for item in batch]
    states = torch.stack([item["state"] for item in batch], dim=0)
    # actions = torch.stack([item["action"] for item in batch], dim=0) # NOTE: UNCOMMENT if NOT using Parallel Action Head implementation
    # action_mask = torch.stack([item["action_mask"] for item in batch], dim=0) # NOTE: UNCOMMENT if NOT using Parallel Action Head implementation
    image_masks = torch.stack([item["image_mask"] for item in batch], dim=0)
    state_mask = torch.stack([item["state_mask"] for item in batch], dim=0)
    embodiment_ids = torch.stack([item["embodiment_id"] for item in batch], dim=0)

    ### ADDED for Parallel Action Head implementation. NOTE COMMENT this block out if NOT using Parallel Action Head implementation
    actions = [item["action"] for item in batch]
    action_mask = [item["action_mask"] for item in batch]
    
    # print(f"In train.py: USING PARALLEL ACTION HEAD LOGIC! Check code here if not using Parallel Action Head implementation.")
    max_horizon = 50# max(a.shape[0] for a in actions) # Obtain the maximum horizon length used (which will be the maximum horizon length found in the actions)
    actions, action_mask = pad_actions(actions, action_mask, max_horizon) # Pad the actions and obtain the corresponding masks specifically only to mask out the extra padded actions
    ###

    return {
        "prompts": prompts,
        "images": images,
        "states": states,
        "actions": actions,
        "action_mask": action_mask,
        "state_mask": state_mask,
        "image_masks": image_masks,
        "embodiment_ids": embodiment_ids
    }

def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step  
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda
    
def setup_logging(log_dir: str) -> str:
    from datetime import datetime
    import logging, os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_path}")
    return log_path

def init_wandb(config: dict, accelerator: Accelerator):

    if accelerator.is_main_process:
        if get_with_warning(config, "disable_wandb", False):
            os.environ["WANDB_MODE"] = "disabled"

        wandb.init(
            project=get_with_warning(config, "wandb_project", "default_run"),
            name=get_with_warning(config, "run_name", "default_run"),
            config=config,
            dir=get_with_warning(config, "save_dir", "checkpoints"),
            mode="offline",
        )

        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

def init_swanlab(config: dict, accelerator: Accelerator):

    if accelerator is None or accelerator.is_main_process:
        swanlab.init(
            project=config.get("wandb_project", "default_run"),
            name=config.get("run_name", "default_run"),
            config=config
        )

def prepare_dataset(config: dict) -> torch.utils.data.Dataset:
    dataset_type = get_with_warning(config, "dataset_type", "lerobot")
    image_size = get_with_warning(config, "image_size", 448)
    max_samples = get_with_warning(config, "max_samples_per_file", None)
    horizon = get_with_warning(config, "horizon", 50)
    nav_horizon = get_with_warning(config, "nav_horizon", 10) ### ADDED for Parallel Action Head implementation
    manip_horizon = get_with_warning(config, "manip_horizon", 50) ### ADDED for Parallel Action Head implementation
    binarize_gripper = get_with_warning(config, "binarize_gripper", False)
    use_augmentation = get_with_warning(config, "use_augmentation", False)
    if dataset_type == "lerobot":
        from lerobot_dataset_pretrain_mp import LeRobotDataset ### EDITED
        import yaml
        with open(config.get("dataset_config_path"), 'r') as f:
            dataset_config = yaml.safe_load(f)

        dataset = LeRobotDataset(
            config=dataset_config,
            image_size=image_size,
            max_samples_per_file=max_samples,
            action_horizon=horizon,
            nav_horizon=nav_horizon, ### ADDED for Parallel Action Head implementation
            manip_horizon=manip_horizon, ### ADDED for Parallel Action Head implementation
            binarize_gripper=binarize_gripper,
            use_augmentation=use_augmentation
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Loaded {len(dataset)} samples from {config['data_paths']} ({dataset_type})")
    return dataset


def prepare_dataloader(dataset, config: dict) -> DataLoader:
    batch_size = get_with_warning(config, "batch_size", 8)
    num_workers = get_with_warning(config, "num_workers", 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Initialized dataloader with batch size {batch_size}")
    return dataloader


def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info(f"[Step {step}] Non-finite detected in {name}")
            return False
    return True

def log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator):
    current_epoch = step / len(dataloader)
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        logging.info(f"[Step {step}] Loss: {loss.item():.4f}")
        wandb.log({
            "step": step,
            "loss": loss.item(),
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            
        })
        # swanlab.log({ # COMMENTED OUT
        #     "step": step,
        #     "loss": loss.item(),
        #     "current_epoch": current_epoch,
        #     "learning_rate": scheduler.get_last_lr()[0],
    
        # })

def save_checkpoint(save_dir, step, model_engine, loss, accelerator, config=None, norm_stats=None):
    tag = f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, tag)

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} exists. Removing before overwrite.")
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()

    client_state = {
        "step": step,
        "best_loss": loss if isinstance(loss, float) else loss.item(),
        "config": config,
    } if accelerator.is_main_process else {} 

    model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
    
    if accelerator.is_main_process:
        if config is not None:
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        if norm_stats is not None:
            norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.json")
            with open(norm_stats_path, "w") as f:
                json.dump(norm_stats, f, indent=2)
                
        checkpoint_meta_path = os.path.join(checkpoint_dir, "checkpoint.json")
        checkpoint_meta = {
            "type": "ds_model",
            "version": 0.0,
            "checkpoints": "mp_rank_00_model_states.pt"
        }
        with open(checkpoint_meta_path, "w") as f:
            json.dump(checkpoint_meta, f, indent=2)
        logging.info(f"[Rank {accelerator.process_index}] Saved checkpoint to {checkpoint_dir}")

def load_checkpoint_with_deepspeed(model_engine, load_dir, accelerator, tag="step_best", load_optimizer_states=True, resume_pretrain=False):

    try:
        load_path, client_state = model_engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=load_optimizer_states and not resume_pretrain,
            load_lr_scheduler_states=load_optimizer_states and not resume_pretrain
        )
        if accelerator.is_main_process:
            logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (including optimizer states)")
        return client_state.get("step", 0), client_state
        
    except Exception as e:
        if accelerator.is_main_process:
            logging.warning(f"World size mismatch detected: {str(e)}")
            logging.warning("Attempting to load only model weights (skipping optimizer states)...")
        try:
            load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=False,
                load_lr_scheduler_states=False
            )
            if accelerator.is_main_process:
                logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (model weights only)")
            return client_state.get("step", 0), client_state
            
        except Exception as e2:
            if accelerator.is_main_process:
                logging.error(f"Failed to load checkpoint even without optimizer states: {str(e2)}")
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {load_dir} with tag {tag}: {str(e2)}")

    

def get_and_clip_grad_norm(accelerator, model, loss, max_norm: float = 1.0):

    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):
       
        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:
 
        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(grad_norms) == 0:
            total_norm = torch.tensor(0.0, device=loss.device)
        else:
            total_norm = torch.norm(torch.stack(grad_norms), 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        clipped_grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(clipped_grad_norms) == 0:
            clipped_norm = torch.tensor(0.0, device=loss.device)
        else:
            clipped_norm = torch.norm(torch.stack(clipped_grad_norms), 2)

    return total_norm, clipped_norm

def build_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith("bias") or ".bias" in n
        is_norm = (p.dim() == 1) or ("norm" in n.lower())
        (no_decay if is_bias or is_norm else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

def train(config):


    # === Set logging ===
    save_dir = get_with_warning(config, "save_dir", "checkpoints")
    log_path = setup_logging(save_dir)

    ### ADDED
    # Set up training metric logging file
    if accelerator.is_main_process:
        with open(TRAINING_METRICS_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            # writer.writerow(["step", "epoch", "mse_loss", "mse_x", "mse_y", "mse_z", "mse_angle1", "mse_angle2", "mse_angle3", "mse_gripper", "mae_loss", "learning_rate", "grad_norm"]) # FOR single action head implementation
            # FOR parallel action head implementation
            writer.writerow(["step", "epoch", "mse_loss", "original_loss", "nav_loss", "manip_loss",
                             "nav_mse_x", "nav_mse_y", "nav_mse_z", "nav_mse_angle1", "nav_mse_angle2", "nav_mse_angle3", "nav_mse_gripper",
                             "manip_mse_x", "manip_mse_y", "manip_mse_z", "manip_mse_angle1", "manip_mse_angle2", "manip_mse_angle3", "manip_mse_gripper",
                             "mae_loss", "nav_samples", "manip_samples", "batch_size", "valid_nav_entries", "valid_manip_entries", "total_entries",
                             "nav_grad_norm", "manip_grad_norm", "grad_dot_product", "grad_cosine_similarity", "pcgrad_projections", "learning_rate", "grad_norm"])
    ###
    
    # === WandB and Swanlab ===
    init_wandb(config, accelerator)
    #init_swanlab(config, accelerator) # COMMENTED OUT

    # === Debug mode ===
    if get_with_warning(config, "debug", False):
        torch.autograd.set_detect_anomaly(True)

    # === Dataset ===
    dataset = prepare_dataset(config)

    # === DataLoader ===
    dataloader = prepare_dataloader(dataset, config)

    # === Model ===
    model = AGVLA(config) ### EDITED
    model.train()
    model.set_finetune_flags()

    lr = get_with_warning(config, "lr", 1e-5)
    wd = get_with_warning(config, "weight_decay", 1e-5)
    optimizer = AdamW(build_param_groups(model, wd), lr=lr)
    if accelerator.is_main_process:
        logging.info(f"Optimizer=AdamW, lr={lr}, weight_decay={wd}")


    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model  
  
    if accelerator.is_main_process:
        logging.info("Initialized with Accelerate")
    
    
    # === Warmup + Cosine Scheduler ===
    max_steps = get_with_warning(config, "max_steps", 1000)
    warmup_steps = get_with_warning(config, "warmup_steps", 300)
    
    # === loss function ===
    loss_fn = nn.MSELoss() 

    # === Checkpoint and save path setup ===
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_checkpoint.pt")
    best_loss = float("inf")
    
    # === Logging and interval settings ===
    log_interval = get_with_warning(config, "log_interval", 100)
    ckpt_interval = get_with_warning(config, "ckpt_interval", 1000)
    max_norm = get_with_warning(config, "grad_clip_norm", 1.0)

    # === Resume training from checkpoint ===
    resume = get_with_warning(config, "resume", False)
    resume_path = get_with_warning(config, "resume_path", None)
    resume_pretrain = get_with_warning(config, "resume_pretrain", False)

    if resume != bool(resume_path):
        raise ValueError("Inconsistent resume configuration: --resume and --resume_path must be set together.")
    
    if resume:
        resume_path = resume_path.rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)

        step, client_state = load_checkpoint_with_deepspeed(
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            load_optimizer_states=True,  
            resume_pretrain=resume_pretrain
        )
        best_loss = client_state.get("best_loss", float("inf"))
        if accelerator.is_main_process:
            logging.info(f"Resuming from {resume_dir}/{resume_tag}, step {step}")
    else:
        step = 0
        if accelerator.is_main_process:
            logging.info("Starting fresh training")

    if resume_pretrain:
        step = 0
        logging.info("Resuming pretraining from scratch, resetting step to 0")

    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=step))


    if accelerator.is_main_process:
        
        inspect_named_submodules({
            "vision_model": model.embedder.model.vision_model,
            "language_model": model.embedder.model.language_model,
            "action_head": model.action_head
        })

    # === Training Loop ===
    while step < max_steps:
        for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break
            prompts = batch["prompts"]
            images_batch = batch["images"]
            image_masks = batch["image_masks"]
            states = batch["states"].to(dtype=torch.bfloat16)
            actions_gt = batch["actions"].to(dtype=torch.bfloat16)
            action_mask = batch["action_mask"]
            state_mask = batch["state_mask"]
            embodiment_ids = batch["embodiment_ids"]
            fused_tokens_list = []

            ### ADDED for discarding datasets that only contain samples from a single dataset type
            nav_mask = embodiment_ids == HABITATSIM_EMBODIMENT_ID
            manip_mask = embodiment_ids == LIBERO_EMBODIMENT_ID

            # Require both task types, otherwise skip this loop (essentially refetching a new batch)
            # NOTE: step does not increment until after a batch is actually trained, so skipping batches won't count as training steps
            if not nav_mask.any() or not manip_mask.any():
                continue
            ###
            
            for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
                fused = model.get_vl_embeddings(images=images, image_mask=image_mask, prompt=prompt, return_cls_only=False)
                fused_tokens_list.append(fused.to(dtype=torch.bfloat16))
            
            fused_tokens = torch.cat(fused_tokens_list, dim=0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                # pred_velocity, noise = model(fused_tokens, state=states, actions_gt=actions_gt, action_mask=action_mask)
                pred_velocity, noise = model(fused_tokens, state=states, actions_gt=actions_gt, action_mask=action_mask, embodiment_ids=embodiment_ids) ### ADDED for ParallelActionHead implementation
                
            target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
            
            assert pred_velocity.shape == target_velocity.shape

            if action_mask.sum() == 0:
                raise ValueError(f"[Step {step}] action_mask.sum() is 0! All actions are masked. "
                            f"This indicates a problem with the data or mask generation. "
                            f"action_mask shape: {action_mask.shape}, "
                            f"action_mask: {action_mask}")
            

            ### DEBUGGING
            # print(f">> In train.py <<")
            # print(f"pred_velocity shape: {pred_velocity.shape}", flush=True)
            # print(f"target_velocity shape: {target_velocity.shape}", flush=True)
            # print(f"action_mask shape: {action_mask.shape}", flush=True)
            # # Print "for this sample, how many timesteps is this action dimension active?"
            # print(f"action_mask sum: {action_mask.sum(dim=1)[:5]}", flush=True) # For HabitatSim data samples, MUST see 240, and for LIBERO samples, MUST see 1200! They can't all be the same, or it means masking is not working (not masking out padded horizon dims)
            # print(f"action_mask sum (num valid dims): {action_mask.sum(dim=2)[:5]}", flush=True) # Prints how many valid dims each timestep has
            # print(f"action_mask sum (total num of active entries): {action_mask.view(action_mask.size(0), -1).sum(dim=1)}", flush=True) # Print total num of active entries per sample
            ###

            action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
            pred_velocity_mask = pred_velocity * action_mask
            target_velocity = target_velocity * action_mask ### ADDED, for masking out roll, pitch and gripper dimensions when training on HabitatSim datasets
            loss = loss_fn(pred_velocity_mask, target_velocity)
            scale_factor = action_mask.numel() / (action_mask.sum() + 1e-8)
            loss = loss * scale_factor
            ### ADDED for PCGrad
            # Create action masks, which selects samples
            nav_mask = embodiment_ids == HABITATSIM_EMBODIMENT_ID # Form NAVIGATION mask
            manip_mask = embodiment_ids == LIBERO_EMBODIMENT_ID # Form MANIPULATION mask

            # Initialise the separate losses
            nav_loss = None
            manip_loss = None

            # Calculate navigation loss:
            if nav_mask.any():
                nav_pred = pred_velocity_mask[nav_mask]
                nav_target = target_velocity[nav_mask]
                nav_action_mask = action_mask[nav_mask]

                nav_loss = loss_fn(nav_pred, nav_target)
                nav_scale_factor = nav_action_mask.numel() / (nav_action_mask.sum() + 1e-8) # Per-task normalisation of loss
                # nav_scale_factor = nav_action_mask.numel() / (action_mask.sum() + 1e-8) # Contribution to global loss - i.e. Decomposition of original loss for this task

                nav_loss = nav_loss * nav_scale_factor

            # Calculate manipulation loss:
            if manip_mask.any():
                manip_pred = pred_velocity_mask[manip_mask]
                manip_target = target_velocity[manip_mask]
                manip_action_mask = action_mask[manip_mask]

                manip_loss = loss_fn(manip_pred, manip_target)
                manip_scale_factor = manip_action_mask.numel() / (manip_action_mask.sum() + 1e-8) # Per-task normalisation of loss
                # manip_scale_factor = manip_action_mask.numel() / (action_mask.sum() + 1e-8) # Contribution to global loss - i.e. Decomposition of original loss for this task

                manip_loss = manip_loss * manip_scale_factor

            # Recombining loss test
            # # print(f"Original loss: {loss}", flush=True)
            # loss = 0.0
            # if nav_loss is not None:
            #     loss = loss + loss_fn(nav_pred, nav_target) * nav_pred.numel() / action_mask.numel()
            # if manip_loss is not None:
            #     loss = loss + loss_fn(manip_pred, manip_target) * manip_pred.numel() / action_mask.numel()

            # loss = loss * scale_factor
            # # print(f"Recombined loss: {loss}", flush=True)
            # # print(f"Nav loss: {nav_loss}", flush=True)
            # # print(f"Manip loss: {manip_loss}", flush=True)
            ###
            
            # === NaN/Inf check ===
            if not check_numerical_stability(
                step,
                states=states,
                actions_gt=actions_gt,
                fused_tokens=fused_tokens,
                pred_velocity=pred_velocity,
                loss=loss
            ):
                continue

            # === Backward and optimizer step ===
            # optimizer.zero_grad(set_to_none=True)
            # accelerator.backward(loss)

            ### ADDED for PCGrad
            model.zero_grad()
            task_losses = [nav_loss, manip_loss] # Initialise list of task losses for individual tasks

            pcgrad_out = pcgrad_backward(accelerator=accelerator, model=model, losses=task_losses, retain_graph=True)
            ###

            ##### DEBUGGING
            # ### DEBUGGING loss shape (for later input into backward())
            # for name, task_loss in [
            #     ("nav_loss", nav_loss),
            #     ("manip_loss", manip_loss),
            # ]:
            #     if task_loss is not None:
            #         print(
            #             f"{name}: "
            #             f"type={type(task_loss)}, "
            #             f"shape={task_loss.shape}, "
            #             f"ndim={task_loss.ndim}, "
            #             f"numel={task_loss.numel()}, "
            #             f"requires_grad={task_loss.requires_grad}",
            #             flush=True,
            #         )
            # ###
            # ### DEBUGGING obtaining two independent task gradients, using DeepSpeed APIs
            # from deepspeed.utils import safe_get_full_grad, safe_set_full_grad

            # nav_grads = {}
            # manip_grads = {}

            # if nav_loss is not None:
            #     model.backward(nav_loss, retain_graph=True)

            #     for name, p in model.named_parameters():
            #         if not p.requires_grad:
            #             continue

            #         g = safe_get_full_grad(p)

            #         if g is not None:
            #             nav_grads[name] = g.detach().clone()

            #     model.zero_grad()

            # if manip_loss is not None:
            #     model.backward(manip_loss)

            #     for name, p in model.named_parameters():
            #         if not p.requires_grad:
            #             continue

            #         g = safe_get_full_grad(p)

            #         if g is not None:
            #             manip_grads[name] = g.detach().clone()

            # for name in nav_grads:

            #     if name not in manip_grads:
            #         continue

            #     g_nav = nav_grads[name].float()
            #     g_manip = manip_grads[name].float()

            #     dot = torch.sum(g_nav * g_manip)

            #     nav_norm = torch.linalg.vector_norm(g_nav)
            #     manip_norm = torch.linalg.vector_norm(g_manip)

            #     cosine = dot / (nav_norm * manip_norm + 1e-12)

            #     print(
            #         name,
            #         "nav_norm =", nav_norm.item(),
            #         "manip_norm =", manip_norm.item(),
            #         "dot =", dot.item(),
            #         "cosine =", cosine.item(),
            #     )
            # ###

            # Split losses + double backwards test
            # print(f"Original loss: {loss}", flush=True)
            original_loss = loss.detach().clone() # Save the original, unsplit loss (for logging/debugging)
            # loss = sum(x for x in (nav_loss, manip_loss) if x is not None) # i.e. nav_loss + manip_loss if both are NOT None, otherwise just whichever exists.
            nav_valid_entries = nav_action_mask.sum() if nav_loss is not None else 0 # Get number of valid nav entries (0 if there are no nav entries)
            manip_valid_entries = manip_action_mask.sum() if manip_loss is not None else 0 # Get number of valid manip entries (0 if there are no manip entries)
            total_valid_entries = nav_valid_entries + manip_valid_entries # Calculate total number of valid entries
            loss = ((nav_loss * nav_valid_entries if nav_loss is not None else 0) + (manip_loss * manip_valid_entries if manip_loss is not None else 0)) / (total_valid_entries + 1e-8) # Calculate original loss
            # print(f"Split combined loss: {loss}", flush=True)
            #####

            # === Clip grad norm ===
            total_norm, clipped_norm = get_and_clip_grad_norm(accelerator, model, loss, max_norm)

            # optimizer.step()
            model.step() ### ADDED
            ### DEBUGGING to check if optimizer actually consumes
            # from deepspeed.utils import (
            #     safe_get_full_grad,
            #     safe_get_full_fp32_param,
            # )

            # debug_param = None
            # debug_name = None

            # for name, p in model.named_parameters():
            #     if name == "module.embedder.model.vision_model.embeddings.class_embedding":
            #         debug_param = p
            #         debug_name = name
            #         break

            # if debug_param is not None:

            #     # --------------------------------------------------
            #     # BEFORE optimizer.step()
            #     # --------------------------------------------------

            #     grad_before = safe_get_full_grad(debug_param)
            #     fp32_before = safe_get_full_fp32_param(debug_param)

            #     print("\nBEFORE optimizer.step():")
            #     print("parameter:", debug_name)

            #     if grad_before is not None:
            #         print("grad norm:",
            #             torch.linalg.vector_norm(grad_before.float()).item())

            #     if fp32_before is not None:
            #         print("FP32 param norm:",
            #             torch.linalg.vector_norm(fp32_before.float()).item())

            #         fp32_before = fp32_before.detach().clone()

            #     param_before = debug_param.detach().clone()

            #     # --------------------------------------------------
            #     # OPTIMIZER STEP
            #     # --------------------------------------------------

            #     # optimizer.step()
            #     model.step()

            #     # --------------------------------------------------
            #     # AFTER optimizer.step()
            #     # --------------------------------------------------

            #     fp32_after = safe_get_full_fp32_param(debug_param)
            #     param_after = debug_param.detach().clone()

            #     print("\nAFTER optimizer.step():")

            #     if fp32_after is not None:
            #         fp32_change = fp32_after.float() - fp32_before.float()

            #         print("FP32 parameter change norm:",
            #             torch.linalg.vector_norm(fp32_change).item())

            #         print("FP32 parameter change max:",
            #             torch.max(torch.abs(fp32_change)).item())

            #     bf16_change = param_after.float() - param_before.float()

            #     print("BF16 parameter change norm:",
            #         torch.linalg.vector_norm(bf16_change).item())

            #     print("BF16 parameter change max:",
            #         torch.max(torch.abs(bf16_change)).item())
            ###
            scheduler.step()
            
            # === Logging ===
            if step % log_interval == 0:
                log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator)

                ### ADDED
                # Calculate per-action dimension losses
                action_dim = actions_gt.shape[-1] # Get the action dimension
                pred_dim = pred_velocity_mask.view(pred_velocity_mask.shape[0], -1, action_dim) # Get the predictions per dimension
                target_dim = target_velocity.view(target_velocity.shape[0], -1, action_dim) # Get the targets per dimension

                # Calculate MAE
                mae = torch.mean(torch.abs(pred_velocity_mask - target_velocity))
                mae = mae * scale_factor # Apply same scaling factor as that applied to MSE loss

                ### FOR SINGLE ACTION HEAD IMPLEMENTATION ###
                # dim_losses = ((pred_dim - target_dim) ** 2).mean(dim=(0, 1)) * scale_factor # Calculate per-dimension losses (same way as calculated previously)

                # # Log training metrics (above is logging in wandb, here it is logging to a csv file)
                # if accelerator.is_main_process:
                #     with open(TRAINING_METRICS_PATH, "a", newline="") as f:
                #         writer = csv.writer(f)
                #         writer.writerow([
                #             step, # Step
                #             (step / len(dataloader)), # = current_epoch
                #             loss.item(), # MSE loss
                #             dim_losses[0].item(), # MSE loss x
                #             dim_losses[1].item(), # MSE loss y
                #             dim_losses[2].item(), # MSE loss z
                #             dim_losses[3].item(), # MSE loss angle1
                #             dim_losses[4].item(), # MSE loss angle2
                #             dim_losses[5].item(), # MSE loss angle3
                #             dim_losses[6].item(), # MSE loss gripper
                #             mae.item(), # MAE
                #             scheduler.get_last_lr()[0], # Learning rate
                #             clipped_norm.item() # Gradient norm
                #         ])
                ######

                ### FOR PARALLEL ACTION HEAD IMPLEMENTATION ###
                # Create action masks, which selects samples
                nav_mask = embodiment_ids == HABITATSIM_EMBODIMENT_ID # Form NAVIGATION mask
                manip_mask = embodiment_ids == LIBERO_EMBODIMENT_ID # Form MANIPULATION mask

                # Calculate scale factor for each action head
                # NOTE that these calculation work because the action mask for EACH navigation/manipulation are the same
                nav_scale_factor = action_mask[nav_mask].numel() / (action_mask[nav_mask].sum() + 1e-8)
                manip_scale_factor = action_mask[manip_mask].numel() / (action_mask[manip_mask].sum() + 1e-8)

                # Compute per-dimension losses for each action head
                # FOR navigation datasets
                if nav_mask.any():
                    nav_pred_dim = pred_dim[nav_mask] # Extract the per-dimension predicted velocities
                    nav_target_dim = target_dim[nav_mask] # Extract the per-dimension target velocities
                    nav_dim_losses = ((nav_pred_dim - nav_target_dim)**2).mean(dim=(0,1)) * nav_scale_factor # Calculate MSE loss (NOTE that target_velocity is already masked above, so MSE calculation will be done with those unecessary dimensions masked aleady)
                else: # IF no navigation datasets
                    nav_dim_losses = torch.zeros(7)

                # FOR manipulation datasets
                if manip_mask.any():
                    manip_pred_dim = pred_dim[manip_mask] # Extract the per-dimension predicted velocities
                    manip_target_dim = target_dim[manip_mask] # Extract the per-dimension target velocities
                    manip_dim_losses = ((manip_pred_dim - manip_target_dim)**2).mean(dim=(0,1)) * manip_scale_factor # Calculate MSE loss
                else: # IF no manipulation datasets
                    manip_dim_losses = torch.zeros(7)

                # Get information relating to number of navigation and manipulation samples in current batch
                num_nav_samples = nav_mask.sum().item() # Get number of navigation samples
                num_manip_samples = manip_mask.sum().item() # Get number of manipulation samples
                batch_size = embodiment_ids.shape[0] # Get batch size (which will be same as number of rows in embodiment_ids)

                num_nav_valid = action_mask[nav_mask].sum().item() # Get number of valid navigation sample entries
                num_manip_valid = action_mask[manip_mask].sum().item() # Get number of valid manipulation sample entries
                total_valid = num_nav_valid + num_manip_valid # Get total number of valid sample entries

                # Log training metrics (above is logging in wandb, here it is logging to a csv file)
                if accelerator.is_main_process:
                    with open(TRAINING_METRICS_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            step, # Step
                            (step / len(dataloader)), # = current_epoch
                            loss.item(), # Overall MSE loss
                            original_loss.item(), # Original overall MSE loss (calculated without splitting)
                            nav_loss.item() if nav_loss is not None else float("nan"), # Navigation loss being passed into PCGrad
                            manip_loss.item() if manip_loss is not None else float("nan"), # Manipulation loss being passed into PCGrad
                            nav_dim_losses[0].item(), # Navigation MSE loss x
                            nav_dim_losses[1].item(), # Navigation MSE loss y
                            nav_dim_losses[2].item(), # Navigation MSE loss z
                            nav_dim_losses[3].item(), # Navigation MSE loss angle1
                            nav_dim_losses[4].item(), # Navigation MSE loss angle2
                            nav_dim_losses[5].item(), # Navigation MSE loss angle3
                            nav_dim_losses[6].item(), # Navigation MSE loss gripper
                            manip_dim_losses[0].item(), # Manipulation MSE loss x
                            manip_dim_losses[1].item(), # Manipulation MSE loss y
                            manip_dim_losses[2].item(), # Manipulation MSE loss z
                            manip_dim_losses[3].item(), # Manipulation MSE loss angle1
                            manip_dim_losses[4].item(), # Manipulation MSE loss angle2
                            manip_dim_losses[5].item(), # Manipulation MSE loss angle3
                            manip_dim_losses[6].item(), # Manipulation MSE loss gripper
                            mae.item(), # MAE
                            num_nav_samples, # Number of navigation samples in current batch
                            num_manip_samples, # Number of manipulation samples in current batch
                            batch_size, # Current batch size
                            num_nav_valid, # Number of valid navigation sample entries
                            num_manip_valid, # Number of valid manipulation sample entries
                            total_valid, # Total number of valid sample entries
                            pcgrad_out['task_grad_norms'][0].item(), # PCGrad navigation norm
                            pcgrad_out['task_grad_norms'][1].item(), # PCGrad manipulation norm
                            (pcgrad_out['dot_product'].item() if pcgrad_out["dot_product"] is not None else float("nan")), # PCGrad dot product
                            (pcgrad_out['cosine_similarity'].item() if pcgrad_out["cosine_similarity"] is not None else float("nan")), # PCGrad cosine similarity
                            pcgrad_out['num_projections'], # PCGrad number of actual projections
                            scheduler.get_last_lr()[0], # Learning rate
                            clipped_norm.item() # Gradient norm
                        ])
                ######
                ###
   
            # === Save best checkpoint ===
            loss_value = loss.item()
            if accelerator.is_main_process:
                is_best = loss_value < best_loss
                if is_best:
                    best_loss = loss_value
                is_best_tensor = torch.tensor(int(is_best), device=accelerator.device)
            else:
                is_best_tensor = torch.tensor(0, device=accelerator.device)
            
            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.broadcast(is_best_tensor, src=0)
            
            if is_best_tensor.item() == 1 and step > 1000:
                accelerator.print("start to save best checkpoint")
                save_checkpoint(
                    save_dir,
                    step="best",
                    model_engine=model_engine,
                    loss=loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=dataset.arm2stats_dict 
                )
                accelerator.print("end to save best checkpoint")
                if accelerator.is_main_process:
                    logging.info(f"Saved best checkpoint at step {step} with loss {loss_value:.6f}")

            step += 1

            # === Save periodic checkpoint ===
            if step % ckpt_interval == 0 and step > 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
                save_checkpoint(save_dir, step=step, model_engine=model_engine, loss=loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict)
         
    # === Save final model ===
    save_checkpoint(save_dir, step="final", model_engine=model_engine, loss=loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict)
    logging.info(f"Final model saved to step_final/")
    logging.info(f"Best checkpoint saved to step_best/ with loss {best_loss:.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train AGVLA") ### EDITED

    # Basic config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--vlm_name", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--action_head", type=str, default="evo1_flowmatching", choices=["evo1_flowmatching", "parallel_action_head"]) ### EDITED
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="lerobot")
    parser.add_argument("--data_paths", type=str, required=False)
    parser.add_argument("--dataset_config_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--binarize_gripper", action="store_true", default=False, help="Whether to binarize gripper state/action (default: False).")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation on images")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)


    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # Resume
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")
   

    # Finetuning
    parser.add_argument("--finetune_vlm", action="store_true")
    parser.add_argument("--finetune_action_head", action="store_true")

    # Misc
    parser.add_argument("--per_action_dim", type=int, default=7)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--nav_horizon", type=int, default=10) ### ADDED for Parallel Action Head implementation, for setting the navigation action head horizon
    parser.add_argument("--manip_horizon", type=int, default=50) ### ADDED for Parallel Action Head implementation for setting the manipulation action head horizon
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    # dropout
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    config = vars(args)

    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)

