#!/bin/bash
#SBATCH --job-name  ft_cnx_full
#SBATCH --time      6:00:00
#SBATCH -c          8
#SBATCH --mem       40G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# ConvNeXt-Tiny 를 Train+Val 로 파인튜닝 (시드 3개) → Test 프레임 확률 ft_convnext_tiny_trval.npz
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export HF_HOME=/home/nxcolelxu/unid/hf_cache
nvidia-smi -L
python finetune_full.py --data_dir /home/nxcolelxu/unid/data/battery_data_light --seeds 0 1 2 2>&1 | tee -a logs/ft_convnext_full.log
