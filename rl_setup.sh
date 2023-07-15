#!/bin/bash
srun --gres=gpu:1 --pty --mem 16G -c 10 -t 240 /bin/bash
source /etc/profile.d/cluster_modules.sh
source /etc/profile.d/modules.sh
module load loadonly/gpu
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
cd /home/kita-h/workspace/pgx
source venv/bin/activate
cd workspace/ppo_bridge/