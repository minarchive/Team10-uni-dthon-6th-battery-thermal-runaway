#!/bin/bash
#SBATCH --time      24:00:00
#SBATCH -c          8
#SBATCH --mem       64G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 사용:  sbatch -J joint_eb0 run_joint2_cnn.sh joint_moe2_eb0 --model efficientnet_b0 --n_unfreeze 7 --lr 3e-4

set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export HF_HOME=/home/nxcolelxu/unid/hf_cache
export HF_HUB_OFFLINE=1
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
OUT=$1; shift
nvidia-smi -L
python joint_moe2_cnn.py --data_dir $DATA --out $OUT "$@" 2>&1 | tee logs/$OUT${SLURM_JOB_NAME:+_$SLURM_JOB_NAME}.log
