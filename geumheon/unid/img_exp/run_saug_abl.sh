#!/bin/bash
#SBATCH --time      6:00:00
#SBATCH -c          16
#SBATCH --mem       96G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 증강 6종을 하나씩만 켜서 비교 (GPU 1개에 6개 프로세스 병렬). 나머지 증강 강도는 0.
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
DATA=${DATA:-/home/nxcolelxu/unid/data/battery_data_light}
nvidia-smi -L
OFF="--tw 0 --drop 0 --gas_sd 0 --atmp_sd 0 --temp_sd 0 --v_sd 0"
COMMON="--data_dir $DATA --n_aug 6 --seeds 0,1,2"
declare -A ON=( [sa_only_tw]="--tw 0.15" [sa_only_drop]="--drop 0.3" [sa_only_gas]="--gas_sd 0.2"
                [sa_only_atmp]="--atmp_sd 0.5" [sa_only_temp]="--temp_sd 0.3" [sa_only_v]="--v_sd 0.005" )
for OUT in "${!ON[@]}"; do
  OMP_NUM_THREADS=2 python sensor_aug.py $COMMON --out $OUT $OFF ${ON[$OUT]} > logs/$OUT.log 2>&1 &
done
wait
grep -h "CV post mean" logs/sa_only_*.log
