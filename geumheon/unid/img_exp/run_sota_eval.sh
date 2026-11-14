#!/bin/bash
#SBATCH --job-name  sota_eval
#SBATCH --time      6:00:00
#SBATCH -c          6
#SBATCH --mem       48G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# SOTA 0.8812 을 우리 CV/Val 규약으로 재측정: LightGBM → (센서만 점수) → EfficientNet → 최종 점수
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export LD_LIBRARY_PATH=/home/nxcolelxu/unid/venv/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
nvidia-smi -L
python sota_eval.py sensor 2>&1 | tee -a logs/sota_eval.log
python sota_eval.py merge  2>&1 | tee -a logs/sota_eval.log
python sota_eval.py vision 2>&1 | tee -a logs/sota_eval.log
python sota_eval.py merge  2>&1 | tee -a logs/sota_eval.log
