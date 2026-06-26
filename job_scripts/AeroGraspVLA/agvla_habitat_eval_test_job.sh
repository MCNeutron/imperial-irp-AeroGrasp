#!/bin/bash
#PBS -N agvla_habitat_eval_test
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=00:10:00

# Go to current working directory
cd $PBS_O_WORKDIR

# GPU check
nvidia-smi

# Ensure headless rendering
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

module list # DEBUG: Check modules loaded

(
# Change working directory to activate uv virtual environment (for openpi)
cd /rds/general/user/ll1225/home/imperial_irp/openpi_test/openpi
source .venv/bin/activate # Activate uv virtual environment

## Change working directory (back, for running the scripts)
cd /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent

### Run model_runner.py ###
#python /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent/online_eval/vla_eval/model_runner.py
python -u /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent/online_eval/vla_eval/model_runner_agvla_test.py

    ## Activate Evo-1 conda env ADD BACK "
    #eval $(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    #conda activate extended_evo1

    #cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/IndoorUAV_eval # Go to scripts dir

    #python model_runner.py # Run the model_runner script
) > debug/agvla_habitat_server.log 2>&1 &

MODEL_PID=$! # Track model_runner.py PID
echo "MODEL_PID: $MODEL_PID" # Show PID
ps -p $MODEL_PID # Show process info

sleep 12 # Gives model time to load

### Run simulator ###
(
    # Activate HabitatSim conda env
    eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    conda activate habitat_sim

    # Run sim_runner.py
    #python /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent/online_eval/vla_eval/sim_runner.py
    python -u /rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/IndoorUAV_eval/sim_runner.py
) & #> debug/agvla_habitat_simulator.log 2>&1 &

SIM_PID=$! # Track sim_runner.py PID
echo "SIM_PID: $SIM_PID" # Show PID
ps -p $SIM_PID # Show process info

sleep 2 # Wait for script to load

### Run controller ###
(
    # Activate HabitatSim conda env
    eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    conda activate habitat_sim

    # Run vla_controller.py
    #python /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent/online_eval/vla_eval/vla_controller.py
    python -u /rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/IndoorUAV_eval/vla_controller.py
) & #> debug/agvla_habitat_controller.log 2>&1 &

CTRL_PID=$! # Track vla_controller.py PID
echo "CTRL_PID: $CTRL_PID" # Show PID
ps -p $CTRL_PID # Show process info

trap "kill $MODEL_PID $SIM_PID $CTRL_PID 2>/dev/null" EXIT # Kill job if failed

wait # Wait for all processes (so job doesn't exit instantly)
