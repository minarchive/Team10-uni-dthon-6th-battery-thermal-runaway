#!/bin/bash
#SBATCH --job-name  oc6_lgb63
#SBATCH --time      12:00:00
#SBATCH -c          16
#SBATCH --mem       60G
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# oc5 R_J6wide 파이프라인, 시계열만 LightGBM(잎 63, 1500, lr 0.02) · 이미지 = 기존 ConvNeXt 확률 0.1. final_oc6_lgb.py 참고
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export LD_LIBRARY_PATH=/home/nxcolelxu/unid/venv/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
GBM_THREADS=16 TAG=oc6_lgb63 OC6_CFGS=L_J6wide_L63 python final_oc6_lgb.py 2>&1 | tee -a logs/oc6_lgb63.log
