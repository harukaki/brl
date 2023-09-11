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
python new_ppo.py EXP_NAME=exp0070 ENT_COEF=0.01 LOAD_INITIAL_MODEL=True INITIAL_MODEL_PATH="sl_log/sl_fair_0_1_illegal_entropy_2/params-400000.pkl" ACTOR_ILLEGAL_ACTION_MASK=True ACTOR_ILLEGAL_ACTION_PENALTY=False NUM_EVAL_ENVS=10000 ILLEGAL_ACTION_L2NORM_COEF=0 UPDATE_EPOCHS=5
