#!/bin/bash
#PBS -N evo1_metaworld_eval_test
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=00:10:00

# Go to current working directory
cd $PBS_O_WORKDIR

# Load GLVND layer
#module purge
module load libglvnd

# Set up headless environment
unset DISPLAY
#export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/nvidia:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/nvidia:/lib64:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
#unset PYOPENGL_PLATFORM
# Force non-device EGL mode
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE=true
# Disable EGL device backend completely
#unset MUJOCO_EGL_DEVICE
##export MUJOCO_EGL_DEVICE=false # Disable EGL device backend (disable device mode)
unset EGL_PLATFORM #=surfaceless

# Set Mesa paths, and display them
#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
echo ">> CHECK IF libEGL.so.1 IS AVAIL <<"
ldconfig -p | grep -E "EGL|GL\.so|OpenGL"

# GPU check
nvidia-smi

# DEBUG: Check if system has EGL available
echo ">> Using find <<"
find /usr -name "libEGL.so*" 2>/dev/null
echo ">> Using ldconfig <<"
ldconfig -p | grep libEGL

module list # DEBUG: Check modules loaded

# Start server
(
    # Activate Evo-1 conda env
    eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    conda activate extended_evo1

    cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/Evo_1 # Go to scripts dir

    python scripts/Evo1_server.py # Run the evo1_server script
    echo "SERVER EXIT CODE: $?"
) > debug/evo1_server.log 2>&1 &

SERVER_PID=$! # Track Evo1 server PID

trap "kill $SERVER_PID 2>/dev/null" EXIT

echo "Waiting for server..."

for i in {1..120}; do
    if ss -lnt | grep -q ":9000"; then
        echo "Server ready"
        break
    fi

    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died"
        cat debug/evo1_server.log
        exit 1
    fi

    sleep 2
done

# Run client
(
    # Activate Evo-1 MetaWorld conda env
    eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    conda activate extended_evo1_metaworld

    export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/nvidia:$LD_LIBRARY_PATH # Prioritise NVIDIA EGL libraries

    ### DEBUGGING
    echo ">> Finding libs <<"
    find $CONDA_PREFIX -name "libOpenGL*"
    find $CONDA_PREFIX -name "libGLX*"
    find $CONDA_PREFIX -name "libEGL*"

    python <<'EOF'
import ctypes

print('===== EGL TEST =====')

try:
    ctypes.CDLL('libEGL.so.1')
    print('SUCCESS: libEGL.so.1')
except Exception as e:
    print('FAILED libEGL.so.1:', e)

try:
    from OpenGL import EGL
    print('SUCCESS: OpenGL.EGL')
except Exception as e:
    print('FAILED OpenGL.EGL:', repr(e))

print('>> EGL init test <<')
from OpenGL import EGL

dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
EGL.eglInitialize(dpy, None, None)

print(EGL.eglQueryString(dpy, EGL.EGL_VENDOR))
EOF

    # DEBUGGING 2
    echo "===== MUJOCO / EGL DEBUG ====="

    nvidia-smi -L

    python - <<'EOF'
import os

print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))
print("MUJOCO_EGL_DEVICE_ID =", os.environ.get("MUJOCO_EGL_DEVICE_ID"))

import mujoco
print("mujoco version:", mujoco.__version__)

try:
    from OpenGL import EGL
    print("OpenGL.EGL import: SUCCESS")
except Exception as e:
    print("OpenGL.EGL import: FAILED")
    print(repr(e))

try:
    from mujoco import egl

    print("Testing EGL device creation...")
    display = egl.create_initialized_egl_device_display()
    print("SUCCESS:", display)

except Exception as e:
    import traceback
    traceback.print_exc()

import os
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))

import gymnasium.envs.mujoco.mujoco_rendering as mr

print("ALL_RENDERERS =", mr._ALL_RENDERERS.keys())

print("MUJOCO_GL set to:", os.environ["MUJOCO_GL"])
EOF

    echo "===== END DEBUG ====="
    ###

    cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/MetaWorld_evaluation # Go to MetaWorld evaluation dir

    python mt50_evo1_client_prompt.py # Run client script
) #> debug/metaworld_client.log 2>&1

CLIENT_EXIT=$? # Track client PID

echo "CLIENT EXIT CODE: $CLIENT_EXIT"
exit $CLIENT_EXIT
