#!/bin/bash
#SBATCH --time      1:00:00
#SBATCH -c          4
#SBATCH --mem       16G
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# fold 병렬 잡이 끝난 뒤 합치기:  sbatch --dependency=afterok:<ids> -J joint_merge run_joint_merge.sh joint_moe_s_u4
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
OUT=$1; shift
python joint_moe2.py --data_dir $DATA --out $OUT --merge_only "$@" 2>&1 | tee logs/${OUT}_merge.log
