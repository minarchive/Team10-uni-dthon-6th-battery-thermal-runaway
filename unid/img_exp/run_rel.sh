#!/bin/bash
#SBATCH --job-name  rel
#SBATCH --time      12:00:00
#SBATCH -c          16
#SBATCH --mem       30G
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# CPU 전용 (GPU 불필요)

set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
python exp_relative.py --data_dir $DATA --out res_relative.json 2>&1 | tee logs/exp_relative.log
