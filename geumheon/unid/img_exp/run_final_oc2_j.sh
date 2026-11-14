#!/bin/bash
#SBATCH --job-name  oc2_j_final
#SBATCH --time      2:00:00
#SBATCH -c          6
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# S_oc2_j_w03 최종 학습만 (Train+Val 5시드) → 제출: 조합 피처 있음(final_oc2_combo) / 없음(final_oc2_gpu)
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
nvidia-smi -L
GBM_BACKEND=xgb_cuda TAG=combo COMBO=1 FINAL_ONLY=S_oc2_j_w03 python final_oc2.py 2>&1 | tee -a logs/oc2_j_final.log
GBM_BACKEND=xgb_cuda TAG=gpu FINAL_ONLY=S_oc2_j_w03 python final_oc2.py 2>&1 | tee -a logs/oc2_j_final.log
