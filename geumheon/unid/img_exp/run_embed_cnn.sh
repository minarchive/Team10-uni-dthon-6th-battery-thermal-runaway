#!/bin/bash
#SBATCH --time      2:00:00
#SBATCH -c          8
#SBATCH --mem       32G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 고정 ImageNet CNN 특징 추출 (ResNet50, EfficientNet-B0)
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export HF_HOME=/home/nxcolelxu/unid/hf_cache HF_HUB_OFFLINE=1
DATA=/home/nxcolelxu/unid/data/battery_data_light
nvidia-smi -L
python embed_cnn.py --data_dir $DATA --model resnet50.a1_in1k --out emb_resnet50.npz 2>&1 | tee logs/embed_resnet50.log
python embed_cnn.py --data_dir $DATA --model efficientnet_b0.ra_in1k --out emb_effb0.npz 2>&1 | tee logs/embed_effb0.log
