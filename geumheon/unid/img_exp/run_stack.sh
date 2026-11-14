#!/bin/bash
#SBATCH --job-name  stack
#SBATCH --time      12:00:00
#SBATCH -c          8
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 사용: sbatch -J stack_effb0 run_stack.sh effb0
#       sbatch -J stack_both  run_stack.sh both
# CUDA 빌드 LightGBM 런타임이 NCCL을 요구하므로 GPU 노드에서 실행

set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
MODE=${1:?Usage: run_stack.sh effb0|both}
# 이 venv의 LightGBM 바이너리는 libnccl.so.2를 요구하지만 클러스터 런타임에 없습니다.
# exp_img.py가 지원하는 XGBoost CUDA 백엔드로 GPU 학습합니다.
export GBM_BACKEND=${GBM_BACKEND:-xgb_cuda}

mkdir -p logs
nvidia-smi -L
echo "GBM_BACKEND=$GBM_BACKEND"
test -s ft_convnext_tiny.npz || { echo "Missing ft_convnext_tiny.npz" >&2; exit 2; }
test -s ft_efficientnet_b0.npz || { echo "Missing ft_efficientnet_b0.npz" >&2; exit 2; }

case "$MODE" in
    effb0)
        python exp_stack_img.py --data_dir "$DATA" --probs ft_efficientnet_b0.npz --out res_stack_effb0.json 2>&1 | tee logs/stack_effb0.log
        ;;
    both)
        python exp_stack_img.py --data_dir "$DATA" --probs ft_convnext_tiny.npz ft_efficientnet_b0.npz --out res_stack_both.json 2>&1 | tee logs/stack_both.log
        ;;
    *) echo "Unknown mode: $MODE (expected effb0 or both)" >&2; exit 2 ;;
esac
