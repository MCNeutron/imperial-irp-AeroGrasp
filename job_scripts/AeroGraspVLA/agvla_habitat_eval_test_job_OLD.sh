#!/bin/bash
#PBS -N agvla_habitat_eval_test
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=00:10:00

# Go to current working directory
cd $PBS_O_WORKDIR

# GPU check
nvidia-smi

# Activate base conda environments
eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"

# Ensure headless rendering
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

module list # DEBUG: Check modules loaded

(
## Change working directory to activate uv virtual environment (for openpi)
#cd /rds/general/user/ll1225/home/imperial_irp/openpi_test/openpi
#source .venv/bin/activate # Activate uv virtual environment

## Change working directory (back, for running the scripts)
#cd /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent

### Run model_runner.py ###
#python /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent/online_eval/vla_eval/model_runner.py

    # Activate Evo-1 conda env
    eval "$(/rds/general/user/ll1225/home/miniconda3/bin/conda shell.bash hook)"
    conda activate extended_evo1

    cd /rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/IndoorUAV_eval # Go to scripts dir

    python model_runner.py # Run the model_runner script
) > debug/agvla_habitat_server.log 2>&1 &
MODEL_PID=$!

sleep 12 # Gives model time to load

(
# Change working directory
cd /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent

### Run simulator ###
conda activate habitat_sim
#python /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent/online_eval/vla_eval/sim_runner.py &
python /rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/IndoorUAV_eval/sim_runner.py &
SIM_PID=$!

sleep 1 # Wait for script to load

### Run controller ###
#python /rds/general/user/ll1225/home/imperial_irp/indoorUAV_test/IndoorUAV-Agent/online_eval/vla_eval/vla_controller.py &
python /rds/general/user/ll1225/home/imperial_irp/extended_evo1/AeroGraspVLA/IndoorUAV_eval/vla_controller.py &
CTRL_PID=$!

# Wait for all processes
wait $SIM_PID
wait $CTRL_PID
) &
HABITAT_PID=$!

wait $MODEL_PID
wait $HABITAT_PID
