#!/bin/bash
#SBATCH --job-name  embed
#SBATCH --time      12:00:00
#SBATCH -c          6
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/embed_%j.out

set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export HF_HOME=/home/nxcolelxu/unid/hf_cache
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
nvidia-smi -L

# 고정 DINOv2 ViT-S/14 (22M) · ViT-B/14 (86M) 임베딩 추출 (학습 없음)
python embed.py --data_dir $DATA --model facebook/dinov2-small --out emb_dinov2s.npz
python embed.py --data_dir $DATA --model facebook/dinov2-base  --out emb_dinov2b.npz --bs 128

# 방식 A 실험 (CPU, LightGBM). 이진 확인 단계(--stage 1)는 건너뜀
python exp_img.py --data_dir $DATA --emb emb_dinov2s.npz --out res_dinov2s.json --stage 2 2>&1 | tee logs/exp_dinov2s.log
python exp_img.py --data_dir $DATA --emb emb_dinov2b.npz --out res_dinov2b.json --stage 2 2>&1 | tee logs/exp_dinov2b.log
