#!/bin/bash
#SBATCH --job-name  p23_spec
#SBATCH --time      2:00:00
#SBATCH -c          16
#SBATCH --mem       48G
#SBATCH --gpus      0
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# SOTA 계열 센서 모델(LightGBM A+C, Extra Trees A+C, 2→3 피처·balanced 변형) OOF 은행 + Train+Val 최종 가중치
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export LD_LIBRARY_PATH=/home/nxcolelxu/unid/venv/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
NJ=16 python p23_spec.py 2>&1 | tee -a logs/p23_spec.log
