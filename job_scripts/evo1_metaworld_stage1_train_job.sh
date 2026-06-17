#!/bin/bash
#PBS -N evo1_metaworld_stage1_train
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=05:00:00

# Go to current working directory
cd $PBS_O_WORKDIR

# Disable SwanLab
export SWANLAB_OFFLINE=true
export SWANLAB_DISABLED=true

# Activate base conda environments
eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"

# Activate Evo-1 conda env
conda activate extended_evo1

# For debugging
nvidia-smi

# Setup variables
export CUDA_VISIBLE_DEVICES=0

# Run training
accelerate launch \
--num_processes 1 \
--num_machines 1 \
--deepspeed_config_file \
ds_config.json scripts/train.py \
--run_name Evo1_metaworld_stage1_train \
--action_head flowmatching \
--use_augmentation \
--lr 1e-5 \
--dropout 0.2 \
--weight_decay 1e-3 \
--batch_size 16 \
--image_size 448 \
--max_steps 5000 \
--log_interval 10 \
--ckpt_interval 2500 \
--warmup_steps 1000 \
--grad_clip_norm 1.0 \
--num_layers 8 \
--horizon 50 \
--finetune_action_head \
--disable_wandb \
--vlm_name OpenGVLab/InternVL3-1B \
--dataset_config_path dataset/config.yaml \
--per_action_dim 24 \
--state_dim 24 \
--save_dir /rds/general/user/ll1225/home/imperial_irp/extended_evo1/checkpoints/hpc_retrain/stage1
