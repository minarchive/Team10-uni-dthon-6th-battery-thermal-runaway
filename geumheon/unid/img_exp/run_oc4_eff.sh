#!/bin/bash
#SBATCH --job-name  oc4_eff
#SBATCH --time      8:00:00
#SBATCH -c          6
#SBATCH --mem       80G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# oc4 R_J6wide 그대로 + 이미지만 v3_sota_et0.8920 EfficientNet-B0(차이 영상)로 교체
# 1) fold별 EfficientNet 학습(CV 5 + Val 1) + 스냅샷 가중치로 Test 확률  2) 센서 체크포인트 재사용해 결합·채점·제출
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export LD_LIBRARY_PATH=/home/nxcolelxu/unid/venv/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
nvidia-smi -L
python effnet_diff_probs.py 2>&1 | tee -a logs/oc4_eff.log
GBM_BACKEND=xgb_cuda python final_oc4_eff.py 2>&1 | tee -a logs/oc4_eff.log
mv final_oc4_eff/submission_R_J6wide_convnext0.1.csv final_oc4_eff/submission_R_J6wide_effdiff0.1.csv
