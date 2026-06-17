#!/bin/bash
#PBS -N evo1_libero_eval_test
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=00:30:00

# Go to current working directory
cd $PBS_O_WORKDIR

# Setup up tmp directory (for use by robosuite)
export TMPDIR=/rds/general/user/ll1225/home/imperial_irp/extended_evo1/tmp
mkdir -p $TMPDIR

### Force single-threaded execution ###
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONUNBUFFERED=1 # Stabilise Python IO buffering
export MALLOC_TRIM_THRESHOLD_=0 # Reduce scheduling noise

### ADDED FOR OSMESA TESTING ###
export TEMP=$TMPDIR
export TMP=$TMPDIR
export XDG_RUNTIME_DIR=$TMPDIR
###

#rm -f /tmp/robosuite.log
#ln -sf $TMPDIR/robosuite.log /tmp/robosuite.log

# Activate base conda environments
eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"

# Setup for headless rendering on HPC
export LIBGL_ALWAYS_SOFTWARE=0
#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:$LD_LIBRARY_PATH
unset DISPLAY
export PYOPENGL_PLATFORM=egl #osmesa #unset PYOPENGL_PLATFORM
export MUJOCO_GL=egl #osmesa
export MUJOCO_LOG_LEVEL=debug
export CUDA_VISIBLE_DEVICES=0
export EGL_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

# Debugging
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 1 > debug/gpu_log.csv &
GPU_LOG_PID=$!

while true; do
    date
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader
    echo "===================="
    sleep 5
done > debug/gpu_process_log.txt &
GPU_PROC_PID=$!

### Start Evo1 server ###
conda activate extended_evo1 # Activate Evo-1 conda env

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:$LD_LIBRARY_PATH

echo "=== SERVER RENDER CHECK ==="
echo "MUJOCO_GL=$MUJOCO_GL"
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"

cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/Evo_1 # Go to scripts dir
echo ">> Starting Evo1 server <<"
python scripts/Evo1_server.py > evo1_server.log 2>&1 & # Run the Evo1_server script
SERVER_PID=$! # Track Evo1 server PID

#trap kill $SERVER_PID" EXIT # End the Evo1 server script for cleanup, even if script exits early from crashing etc.
trap "kill $SERVER_PID $GPU_LOG_PID $GPU_PROC_PID 2>/dev/null" EXIT

# Wait for server to initialise
echo "Waiting for server to open port 9000..."

# Real server startup readiness check
for i in {1..60}; do
    if ss -lnt | grep -q ":9000"; then
        echo "Server is up!"
        break
    fi

    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server died early. Log:"
        cat evo1_server.log
        exit 1
    fi

    sleep 2
done

# Debug Line
echo "Checking process + ports:"
ps -fp $SERVER_PID
ss -lntp | grep 9000 || echo "Port 9000 not open yet"

### Run LIBERO evaluation ###
conda activate extended_evo1_libero # activate Evo-1 LIBERO conda env

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:$LD_LIBRARY_PATH

echo "=== CLIENT RENDER CHECK ==="
echo "MUJOCO_GL=$MUJOCO_GL"
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"

# Setup for headless rendering on HPC
#export LIBGL_ALWAYS_SOFTWARE=1
#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:$LD_LIBRARY_PATH
#unset DISPLAY
#export PYOPENGL_PLATFORM=egl #osmesa #unset PYOPENGL_PLATFORM
#export MUJOCO_GL=egl #osmesa
#export CUDA_VISIBLE_DEVICES=0
#export EGL_VISIBLE_DEVICES=0
#export MUJOCO_EGL_DEVICE_ID=0

# DEBUG Block #
echo "=== DEBUG BLOCK ==="
echo "MUJOCO_GL=$MUJOCO_GL"
python -c "import mujoco; print('mujoco ok')"
python -c "import OpenGL; print('opengl ok')"
python -c "import robosuite; print('robosuite ok')"

cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/LIBERO_evaluation # Go to LIBERO evaluation dir

# Ensure correct server port is set
export SERVER_URL=ws://127.0.0.1:9000
echo "SERVER_URL=$SERVER_URL"

# Force a clean, deterministic single-GPU + EGL rendering setup (to ensure matching GPU ID format)
#unset CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0
#export MUJOCO_GL=egl #osmesa
#export PYOPENGL_PLATFORM=egl #osmesa
#export EGL_VISIBLE_DEVICES=0
#export MUJOCO_EGL_DEVICE_ID=0

python libero_client_4tasks.py
