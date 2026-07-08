#!/bin/bash
#PBS -N convert_habitatsim_to_lerobot_hm3d_1
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=10:00:00

# Go to current working directory
cd $PBS_O_WORKDIR

# Activate lerobot_dataset conda env
eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
conda activate lerobot_dataset

# Go to converter script directory
cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/helpers

# Run script with desired group scene to convert
python -u habitatsim_to_lerobot_dataset_converter.py --scene_group hm3d_1

# Print completion message
echo "Conversion finished: $(date)"
