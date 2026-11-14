#!/bin/bash
#SBATCH --time      4:00:00
#SBATCH -c          8
#SBATCH --mem       48G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 맥락 피처 3종 (A 궤적, B 실험 기준 상대값, C 프로토타입·kNN) 단독/2개/3개 조합 비교. GBM=XGBoost CUDA, kNN=torch GPU
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export GBM_BACKEND=xgb_cuda
nvidia-smi -L
python exp_ctx.py "$@" 2>&1 | tee logs/ctx.log
