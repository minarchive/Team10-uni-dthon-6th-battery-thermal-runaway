#!/bin/bash
#SBATCH --job-name  oc3_f
#SBATCH --time      6:00:00
#SBATCH -c          6
#SBATCH --mem       40G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
nvidia-smi -L
GBM_BACKEND=xgb_cuda TAG=oc3f OC3_CFGS=J6thin3,J6thin3_ex50 python final_oc3.py 2>&1 | tee -a logs/final_oc3_f.log
