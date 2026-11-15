#!/bin/bash
#SBATCH --job-name  oc5_b127
#SBATCH --time      16:00:00
#SBATCH -c          6
#SBATCH --mem       60G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# oc4 R_J6wide 확장: 트리 키우기(_B63/_B127) + OOD 관련 피처(OOD=0). final_oc5.py 설명 참고
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
nvidia-smi -L
GBM_BACKEND=xgb_cuda TAG=oc5_base127 OOD=0 OC5_CFGS=R_J6wide_B127 python final_oc5.py 2>&1 | tee -a logs/oc5_b127.log
