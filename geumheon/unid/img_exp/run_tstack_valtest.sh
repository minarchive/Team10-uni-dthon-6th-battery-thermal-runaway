#!/bin/bash
#SBATCH --job-name  tstack_valtest
#SBATCH --time      8:00:00
#SBATCH -c          6
#SBATCH --mem       80G
#SBATCH --gpus      1
#SBATCH --constraint ada
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 시간 스택 EfficientNet-B0 (RGB 24채널: 현재 RGB + 첫/10/20/30/60/120/300초 전 RGB 차이). temporal_stack.py 설명 참고
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
python temporal_stack.py val 2>&1 | tee -a logs/tstack.log
python temporal_stack.py test 2>&1 | tee -a logs/tstack.log
