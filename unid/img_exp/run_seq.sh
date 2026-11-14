#!/bin/bash
#SBATCH --job-name=seq
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --constraint=ada
#SBATCH --output=/home/nxcolelxu/unid/img_exp/logs/seq_%j.out

set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export GBM_BACKEND=xgb_cuda
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
mkdir -p logs
nvidia-smi -L
echo "GBM_BACKEND=$GBM_BACKEND"
python seq_experiments.py --data_dir "$DATA" --emb emb_dinov2s.npz --out_dir . 2>&1 | tee logs/seq.log
