#!/bin/bash
#SBATCH --time      24:00:00
#SBATCH -c          8
#SBATCH --mem       40G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 사용:  sbatch -J ft_convnext run_ft.sh convnext_tiny
#        sbatch -J ft_effb0    run_ft.sh efficientnet_b0 --lr 3e-4 --bs 128
# 스태킹은 run_stack.sh 를 --dependency=afterok:<두 잡> 으로 따로 제출

set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export HF_HOME=/home/nxcolelxu/unid/hf_cache
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
MODEL=$1; shift
nvidia-smi -L

# 소형 백본 파인튜닝
python finetune.py --data_dir $DATA --model $MODEL --out ft_$MODEL "$@" 2>&1 | tee logs/ft_$MODEL.log
