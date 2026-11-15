#!/bin/bash
#SBATCH --job-name  eff_oof
#SBATCH --time      4:00:00
#SBATCH -c          6
#SBATCH --mem       80G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# SOTA EfficientNet-B0(차이 영상) fold 확률: Train 5-fold OOF + Train 전체→Val (앙상블 CV용). Test 는 스냅샷 가중치 확률이 이미 있음
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export LD_LIBRARY_PATH=/home/nxcolelxu/unid/venv/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
python effnet_diff_probs.py 2>&1 | tee -a logs/eff_oof.log
