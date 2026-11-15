#!/bin/bash
#SBATCH --job-name  press_hyp5
#SBATCH --time      8:00:00
#SBATCH -c          8
#SBATCH --mem       64G
#SBATCH --gpus      0
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 가설 5: 시작 압력이 높은 실험은 압력 상승폭이 작다 → ref_AC(SOTA) vs C_rel(기준점만 뺌) vs C_rel_p0(p0 피처 추가), 이미지 ConvNeXt α=0.1
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export PRESS_VARIANTS=ref_AC,C_rel,C_rel_p0
python press_exp.py hyp    2>&1 | tee -a logs/press_hyp5.log
python press_exp.py sensor 2>&1 | tee -a logs/press_hyp5.log
python press_exp.py merge  2>&1 | tee -a logs/press_hyp5.log
