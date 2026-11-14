#!/bin/bash
#SBATCH --job-name  final_mm
#SBATCH --time      8:00:00
#SBATCH -c          6
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export GBM_BACKEND=xgb_cuda TAG=gpu
nvidia-smi -L
python final_mm.py 2>&1 | tee -a logs/final_mm_gpu.log
