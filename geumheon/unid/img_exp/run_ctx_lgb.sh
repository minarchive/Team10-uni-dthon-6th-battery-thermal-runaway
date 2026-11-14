#!/bin/bash
#SBATCH --time      3:00:00
#SBATCH -c          8
#SBATCH --mem       48G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# LightGBM(CPU) GBM 시드 1개로 v3 vs v3+A(궤적 맥락) 비교.  사용: sbatch -J ctxl_s0 run_ctx_lgb.sh 0
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export LD_LIBRARY_PATH=/home/nxcolelxu/unid/venv/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
SEED=$1
export GBM_BACKEND=lgbm_cpu GBM_THREADS=8 CTX_SEEDS=$SEED CTX_TAG=_lgb_s$SEED
python exp_ctx.py v3 A 2>&1 | tee logs/ctx_lgb_s$SEED.log
