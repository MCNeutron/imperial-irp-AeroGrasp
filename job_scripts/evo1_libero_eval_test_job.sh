#!/bin/bash
#PBS -N evo1_libero_eval_test
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=01:00:00

# Go to current working directory
cd $PBS_O_WORKDIR

# Setup up tmp directory (for use by robosuite)
export TMPDIR=/rds/general/user/ll1225/home/imperial_irp/extended_evo1/tmp
mkdir -p $TMPDIR

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
unset DISPLAY
export PYOPENGL_PLATFORM=osmesa #egl #unset PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa #egl
export CUDA_VISIBLE_DEVICES=0
#export EGL_VISIBLE_DEVICES=0
#export MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:$LD_LIBRARY_PATH

### Start Evo1 server ###
conda activate extended_evo1 # Activate Evo-1 conda env

cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/Evo_1 # Go to scripts dir
echo ">> Starting Evo1 server <<"
python scripts/Evo1_server.py > evo1_server.log 2>&1 & # Run the Evo1_server script
SERVER_PID=$! # Track Evo1 server PID

trap "kill $SERVER_PID" EXIT # End the Evo1 server script for cleanup, even if script exits early from crashing etc.

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

# Setup for headless rendering on HPC
unset DISPLAY
export PYOPENGL_PLATFORM=osmesa #egl #unset PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa #egl
export CUDA_VISIBLE_DEVICES=0
#export EGL_VISIBLE_DEVICES=0
#export MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:$LD_LIBRARY_PATH

cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/LIBERO_evaluation # Go to LIBERO evaluation dir

# Ensure correct server port is set
export SERVER_URL=ws://127.0.0.1:9000
echo "SERVER_URL=$SERVER_URL"

# Force a clean, deterministic single-GPU + EGL rendering setup (to ensure matching GPU ID format)
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa #egl
export PYOPENGL_PLATFORM=osmesa #egl
#export EGL_VISIBLE_DEVICES=0
#export MUJOCO_EGL_DEVICE_ID=0

python libero_client_4tasks.py
