#!/bin/bash
#PBS -N evo1_libero_eval_test_simplified_egl
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=00:10:00

# Go to current working directory
cd $PBS_O_WORKDIR

# Set up tmp directory (for use by robosuite)
export TMPDIR=/rds/general/user/ll1225/home/imperial_irp/extended_evo1/tmp
mkdir -p "$TMPDIR"

# Required for headless HPC rendering
unset DISPLAY
#export LD_PRELOAD=/usr/lib64/libEGL_nvidia.so.0 # Force NVIDIA EGL at load time
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/nvidia:$LD_LIBRARY_PATH # Force NVIDIA driver libraries to take priority over Conda Mesa EGL
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
#export NVIDIA_VISIBLE_DEVICES=0 # CHECK IF THIS IS NEEDED

# GPU check
nvidia-smi

# Start server
(
    # Activate Evo-1 conda env
    eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    conda activate extended_evo1

    export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/nvidia:$LD_LIBRARY_PATH # Reassert Nvidia-first order after conda activation

    echo ">> SERVER DEBUG <<"
    ### DEBUG to check env variables survived conda activation, and which EGL is being used
    # Full LD_LIBRARY_PATH debug
    echo ">> CLIENT ENV VARIABLES <<"
    echo "MUJOCO_GL=$MUJOCO_GL"
    echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
    echo "CONDA_PREFIX=$CONDA_PREFIX"

    echo ""
    echo ">> LD_LIBRARY_PATH (raw) <<"
    echo "$LD_LIBRARY_PATH"

    echo ""
    echo ">> LD_LIBRARY_PATH (split priority order) <<"
    echo "$LD_LIBRARY_PATH" | tr ':' '\n'

    echo ""
    echo ">> EGL/GL LIBRARY RESOLUTION CHECK <<"
    ldconfig -p | grep -i egl || echo "NO LDCONFIG EGL ENTRIES"

    # Check for what egl is actually being used
    echo ""
    echo ">> RESOLVED libEGL (via ldd check) <<"
    python -c "import ctypes; print(ctypes.CDLL('libEGL.so.1'))" 2>&1

    echo ""
    echo ">> RESOLVED libGL (via ldd check) <<"
    python -c "import ctypes; print(ctypes.CDLL('libGL.so.1'))" 2>&1

    # Force print which egl file is loaded
    python -c "import ctypes; egl = ctypes.CDLL('libEGL.so.1'); print('libEGL loaded:', egl._name)"

    # Detect Mesa vs NVIDIA egl
    echo ""
    echo ">> CHECK FOR MESA EGL IN CONDA <<"
    ls $CONDA_PREFIX/lib/libEGL* 2>/dev/null || echo "No Conda EGL"

    echo ""
    echo ">> CHECK FOR NVIDIA EGL IN SYSTEM PATH <<"
    find /usr /lib64 -name "libEGL_nvidia.so*" 2>/dev/null | head

    # Check egl renderer
    echo ""
    python -c "from OpenGL import GL; print('Renderer:', GL.glGetString(GL.GL_RENDERER)); print('Vendor:', GL.glGetString(GL.GL_VENDOR))"

    echo ""
    echo ">> EGL LOADER DEBUG <<"
    LD_DEBUG=libs python -c "import ctypes; ctypes.CDLL('libEGL.so.1')" 2>&1 | grep -Ei 'EGL|nvidia|mesa|libEGL'

    python -c "from OpenGL import EGL; dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY); EGL.eglInitialize(dpy, None, None); print('VENDOR:', EGL.eglQueryString(dpy, EGL.EGL_VENDOR)); print('VERSION:', EGL.eglQueryString(dpy, EGL.EGL_VERSION))"
    ###

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

    export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/nvidia:$LD_LIBRARY_PATH # Reassert Nvidia-first order after conda activation

    echo ">> CLIENT DEBUG <<"
    ### DEBUG to check env variables survived conda activation, and which EGL is being used
    # Full LD_LIBRARY_PATH debug
    echo ">> CLIENT ENV VARIABLES <<"
    echo "MUJOCO_GL=$MUJOCO_GL"
    echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
    echo "CONDA_PREFIX=$CONDA_PREFIX"

    echo ""
    echo ">> LD_LIBRARY_PATH (raw) <<"
    echo "$LD_LIBRARY_PATH"

    echo ""
    echo ">> LD_LIBRARY_PATH (split priority order) <<"
    echo "$LD_LIBRARY_PATH" | tr ':' '\n'

    echo ""
    echo ">> EGL/GL LIBRARY RESOLUTION CHECK <<"
    ldconfig -p | grep -i egl || echo "NO LDCONFIG EGL ENTRIES"

    # Check for what egl is actually being used
    echo ""
    echo ">> RESOLVED libEGL (via ldd check) <<"
    python -c "import ctypes; print(ctypes.CDLL('libEGL.so.1'))" 2>&1

    echo ""
    echo ">> RESOLVED libGL (via ldd check) <<"
    python -c "import ctypes; print(ctypes.CDLL('libGL.so.1'))" 2>&1

    # Force print which egl file is loaded
    python -c "import ctypes; egl = ctypes.CDLL('libEGL.so.1'); print('libEGL loaded:', egl._name)"

    # Detect Mesa vs NVIDIA egl
    echo ""
    echo ">> CHECK FOR MESA EGL IN CONDA <<"
    ls $CONDA_PREFIX/lib/libEGL* 2>/dev/null || echo "No Conda EGL"

    echo ""
    echo ">> CHECK FOR NVIDIA EGL IN SYSTEM PATH <<"
    find /usr /lib64 -name "libEGL_nvidia.so*" 2>/dev/null | head

    # Check egl renderer
    echo ""
    python -c "from OpenGL import GL; print('Renderer:', GL.glGetString(GL.GL_RENDERER)); print('Vendor:', GL.glGetString(GL.GL_VENDOR))"

    echo ""
    echo ">> EGL LOADER DEBUG <<"
    LD_DEBUG=libs python -c "import ctypes; ctypes.CDLL('libEGL.so.1')" 2>&1 | grep -Ei 'EGL|nvidia|mesa|libEGL'

    # Verify
    python -c "from OpenGL import EGL; dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY); EGL.eglInitialize(dpy, None, None); print('VENDOR:', EGL.eglQueryString(dpy, EGL.EGL_VENDOR)); print('VERSION:', EGL.eglQueryString(dpy, EGL.EGL_VERSION))"
    ###

    cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/LIBERO_evaluation # Go to LIBERO evaluation dir

    export SERVER_URL=ws://127.0.0.1:9000 # Ensure correct server port is set

    #python libero_client_4tasks.py # Run client script
    LD_DEBUG=libs python -u libero_client_4tasks.py 2>&1 | grep -Ei "EGL|nvidia|mesa"
)
