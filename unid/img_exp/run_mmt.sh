#!/bin/bash
#SBATCH --time      8:00:00
#SBATCH -c          8
#SBATCH --mem       48G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# Multi-modal Transformer Early Fusion.  사용: sbatch -J mmt_r50 run_mmt.sh mmt_r50 --img resnet50
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
OUT=$1; shift
nvidia-smi -L
python mm_transformer.py --data_dir /home/nxcolelxu/unid/data/battery_data_light --out $OUT "$@" 2>&1 | tee logs/$OUT.log
