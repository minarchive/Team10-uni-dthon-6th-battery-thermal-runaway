#!/bin/bash
#SBATCH --job-name  bench
#SBATCH --time      01:00:00
#SBATCH -c          8
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
nvidia-smi -L
python bench_gbm.py /home/nxcolelxu/unid/data/battery_data_light
