#!/bin/bash
#SBATCH --time      24:00:00
#SBATCH -c          8
#SBATCH --mem       48G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 사용:  sbatch -J ftd_s_u4  run_ft_dino.sh facebook/dinov2-small 4  ft_ord_dinov2s_u4
#        sbatch -J ftd_b_u4  run_ft_dino.sh facebook/dinov2-base  4  ft_ord_dinov2b_u4
#        sbatch -J ftd_s_u12 run_ft_dino.sh facebook/dinov2-small 12 ft_ord_dinov2s_u12

set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export HF_HOME=/home/nxcolelxu/unid/hf_cache
export HF_HUB_OFFLINE=1
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
MODEL=$1; NU=$2; OUT=$3; shift 3
nvidia-smi -L
python finetune_dino.py --data_dir $DATA --model $MODEL --n_unfreeze $NU --out $OUT "$@" 2>&1 | tee logs/$OUT.log
