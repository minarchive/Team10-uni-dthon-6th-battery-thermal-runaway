#!/bin/bash
#SBATCH --time      6:00:00
#SBATCH -c          8
#SBATCH --mem       48G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 사용:  sbatch -J sa_aug run_saug.sh sa_aug --n_aug 6 --seeds 0,1,2
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
OUT=$1; shift
nvidia-smi -L
python sensor_aug.py --data_dir $DATA --out $OUT "$@" 2>&1 | tee logs/$OUT.log
