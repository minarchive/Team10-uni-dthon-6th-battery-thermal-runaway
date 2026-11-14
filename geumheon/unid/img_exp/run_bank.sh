#!/bin/bash
#SBATCH --time      06:00:00
#SBATCH -c          8
#SBATCH --mem       30G
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 사용: sbatch -J bank_cos run_bank.sh v3+cos   (CPU 전용)
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export OMP_NUM_THREADS=8
python oof_bank.py --data_dir /home/nxcolelxu/unid/data/battery_data_light --cfg "$1"
