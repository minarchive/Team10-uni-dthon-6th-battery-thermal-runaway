#!/bin/bash
#SBATCH --time      4:00:00
#SBATCH -c          8
#SBATCH --mem       48G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 385피처 결과와 같은 평가 방식(Train+Val 97실험 5-fold, 500행 캡, prefix Viterbi)으로 v3 / v3-gas 기준선
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export GBM_BACKEND=xgb_cuda
nvidia-smi -L
python exp_97.py --cap_first 1 --out res_97_capfirst.json 2>&1 | tee logs/exp_97_capfirst.log
python exp_97.py --cap_first 0 --out res_97_fullfeat.json 2>&1 | tee logs/exp_97_fullfeat.log
