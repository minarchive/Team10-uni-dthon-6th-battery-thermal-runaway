#!/bin/bash
#SBATCH --job-name=infer_both
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --constraint=ada
#SBATCH --output=/home/nxcolelxu/unid/img_exp/logs/%x_%j.out

set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export GBM_BACKEND=xgb_cuda
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
MODEL=sub_ts_both_trval.model.joblib
mkdir -p logs
nvidia-smi -L
echo "GBM_BACKEND=$GBM_BACKEND"
if [ -s "$MODEL" ]; then
  python make_submission.py --data_dir "$DATA" \
    --probs ft_convnext_tiny.npz ft_efficientnet_b0.npz \
    --out sub_ts_both_trval.csv --model_in "$MODEL" 2>&1 | tee logs/infer_both.log
else
  python make_submission.py --data_dir "$DATA" \
    --probs ft_convnext_tiny.npz ft_efficientnet_b0.npz \
    --out sub_ts_both_trval.csv --model_out "$MODEL" --train_val 2>&1 | tee logs/infer_both.log
fi
