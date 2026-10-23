#!/bin/bash
#SBATCH --job-name=mbm-mammalia
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=logs/mbm-mammalia-%j.out
#SBATCH --error=logs/mbm-mammalia-%j.err

# Stop on errors and enter the project directory
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

# Activate the Python environment
source calmip/env.sh

# Record software modules and GPU information in the logs
module list
nvidia-smi

# Launch training with Kairos GPU placement
export BIND_CPUNODE=0
srun --unbuffered gbind python -u main.py