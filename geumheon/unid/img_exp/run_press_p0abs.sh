#!/bin/bash
#SBATCH --job-name  press_p0abs
#SBATCH --time      8:00:00
#SBATCH -c          8
#SBATCH --mem       64G
#SBATCH --gpus      0
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# 압력 절대값 유지 + p0 추가 (차이값 prel_* 없음): A_p0, AC_p0 vs ref_AC·C_rel_p0, 이미지 ConvNeXt α=0.1
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export PRESS_VARIANTS=ref_AC,C_rel_p0,A_p0,AC_p0
python press_exp.py sensor 2>&1 | tee -a logs/press_p0abs.log
python press_exp.py merge  2>&1 | tee -a logs/press_p0abs.log
