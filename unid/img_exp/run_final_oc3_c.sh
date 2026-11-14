#!/bin/bash
#SBATCH --job-name  oc3_c
#SBATCH --time      6:00:00
#SBATCH -c          6
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
nvidia-smi -L
GBM_BACKEND=xgb_cuda TAG=oc3c OC3_CFGS=J12wide,J12xwide python final_oc3.py 2>&1 | tee -a logs/final_oc3_c.log
