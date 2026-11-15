#!/bin/bash
#SBATCH --job-name  knn_exp
#SBATCH --time      8:00:00
#SBATCH -c          16
#SBATCH --mem       64G
#SBATCH --gpus      0
#SBATCH --output    /home/nxcolelxu/unid/img_exp/logs/%x_%j.out
# k-최근접 실험 모델 LOO (97개): global / local_k1·5·15·30 / wk15, SOTA LightGBM A·C 분기, ConvNeXt 0.1
set -eo pipefail
cd /home/nxcolelxu/unid/img_exp
source /home/nxcolelxu/unid/venv/bin/activate
export KNN_PAR=4 KNN_THREADS=4
python knn_exp.py run   2>&1 | tee -a logs/knn_exp.log
python knn_exp.py score 2>&1 | tee -a logs/knn_exp.log
