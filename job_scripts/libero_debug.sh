#!/bin/bash
#PBS -N libero_debug
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=00:20:00

# Go to current working directory
cd $PBS_O_WORKDIR

# Activate base conda environments
eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
conda activate extended_evo1_libero

echo "=== GPU CHECK ==="
nvidia-smi

echo "=== EGL IMPORT TEST ==="
python -c "from OpenGL import EGL; print('EGL import OK')"

echo "=== MUJOCO IMPORT TEST ==="
python -c "import mujoco; print('MuJoCo import OK')"
