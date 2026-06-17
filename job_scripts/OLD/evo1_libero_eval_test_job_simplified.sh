#!/bin/bash
#PBS -N evo1_libero_eval_test_simplified
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=00:30:00

# Go to current working directory
cd $PBS_O_WORKDIR

# Set up tmp directory (for use by robosuite)
export TMPDIR=/rds/general/user/ll1225/home/imperial_irp/extended_evo1/tmp
mkdir -p "$TMPDIR"

# Required for headless HPC rendering
unset DISPLAY
#export MUJOCO_GL=osmesa #egl
#export PYOPENGL_PLATFORM=osmesa #egl
export CUDA_VISIBLE_DEVICES=0
#export NVIDIA_VISIBLE_DEVICES=0 # CHECK IF THIS IS NEEDED

### DEBUGGING: See what library paths Python is using
echo "=== LD_LIBRARY_PATH ==="
echo $LD_LIBRARY_PATH

echo "=== CONDA_PREFIX ==="
echo $CONDA_PREFIX
###

# Start server
(
    # Activate Evo-1 conda env
    eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    conda activate extended_evo1

    # DEBUG to check env variables survived conda activation
    echo ">> SERVER ENV VARIABLES <<"
    echo "MUJOCO_GL=$MUJOCO_GL"
    echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
    echo "CONDA_PREFIX=$CONDA_PREFIX"

    # Enforce OSMesa (done here as sometimes conda activation can override globally defined OpenGL-related env vars)
    #export LD_PRELOAD=$CONDA_PREFIX/lib/libGLdispatch.so.0:$CONDA_PREFIX/lib/libOSMesa.so.8 # Add missing GLVND path (the libGL dispatch layer)
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # Set current conda env library folder
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    export PYOPENGL_LAZY_LOAD=0 # Makes PyOpenGL load bindings more eagerly at import/startup time instead of waiting until a function is accessed
    export LIBGL_ALWAYS_SOFTWARE=1 # Force Mesa/OpenGL to use a software renderer instead of GPU hardware acceleration
    python -c 'import mujoco; print("MuJoCo OK")' # DEBUG: Verify MuJoCo backend explicitly, to catch silent EGL fallback cases
    test -f $CONDA_PREFIX/lib/libOSMesa.so.8 || echo "MISSING OSMESA" # DEBUG: Check if OSMesa file exists
    python -c "import os; print(os.environ['LD_LIBRARY_PATH'])" # Check where libOSMesa is being resolved from

    cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/Evo_1 # Go to scripts dir

    python -u scripts/Evo1_server.py # Run the evo1_server script
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
    # Activate Evo-1 LIBERO conda env
    eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    conda activate extended_evo1_libero

    # DEBUG to check env variables survived conda activation
    echo ">> CLIENT ENV VARIABLES <<"
    echo "MUJOCO_GL=$MUJOCO_GL"
    echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
    echo "CONDA_PREFIX=$CONDA_PREFIX"

    # Enforce OSMesa (done here as sometimes conda activation can override globally defined OpenGL-related env vars)
    #export LD_PRELOAD=$CONDA_PREFIX/lib/libGLdispatch.so.0:$CONDA_PREFIX/lib/libOSMesa.so.8 # Add missing GLVND path (the libGL dispatch layer)
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # Set current conda env library folder
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    export PYOPENGL_LAZY_LOAD=0 # Makes PyOpenGL load bindings more eagerly at import/startup time instead of waiting until a function is accessed
    export LIBGL_ALWAYS_SOFTWARE=1 # Force Mesa/OpenGL to use a software renderer instead of GPU hardware acceleration
    python -c 'import mujoco; print("MuJoCo OK")' # DEBUG: Verify MuJoCo backend explicitly, to catch silent EGL fallback cases
    test -f $CONDA_PREFIX/lib/libOSMesa.so.8 || echo "MISSING OSMESA" # DEBUG: Check if OSMesa file exists
    python -c "import os; print(os.environ['LD_LIBRARY_PATH'])" # Check where libOSMesa is being resolved from

    cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/LIBERO_evaluation # Go to LIBERO evaluation dir

    export SERVER_URL=ws://127.0.0.1:9000 # Ensure correct server port is set

    python libero_client_4tasks.py # Run client script
)
