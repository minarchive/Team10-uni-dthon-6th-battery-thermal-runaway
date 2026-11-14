#!/bin/bash
#SBATCH --job-name  oc4
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
GBM_BACKEND=xgb_cuda TAG=gpu python final_oc4.py 2>&1 | tee -a logs/final_oc4.log
