#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem 16G
#SBATCH -c 10
#SBATCH -t 720
source /etc/profile.d/cluster_modules.sh
source /etc/profile.d/modules.sh
module load loadonly/gpu
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
source venv/bin/activate
cd workspace/ppo_bridge
python new_ppo.py ENT_COEF=0.000001 EXP_NAME=exp0024 UPDATE_EPOCHS=4
