#!/bin/bash
#SBATCH --time 4:00:00
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH --gpus 1
#SBATCH --constraint ada
#SBATCH --output /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export GBM_BACKEND=xgb_cuda
python exp_causal.py 2>&1 | tee logs/causal_$SLURM_JOB_ID.log
