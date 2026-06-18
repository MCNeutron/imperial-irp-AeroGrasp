#!/bin/bash
#PBS -N evo1_libero_eval_test
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=00:30:00

# Go to current working directory
cd $PBS_O_WORKDIR

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
    # Activate Evo-1 LIBERO conda env
    eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    conda activate extended_evo1_libero

    # Set up headless environment
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export CUDA_VISIBLE_DEVICES=0

    cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/Evo-1/LIBERO_evaluation # Go to LIBERO evaluation dir

    python libero_client_4tasks.py # Run client script
) #> debug/libero_client.log 2>&1

CLIENT_EXIT=$? # Track client PID

echo "CLIENT EXIT CODE: $CLIENT_EXIT"
exit $CLIENT_EXIT
