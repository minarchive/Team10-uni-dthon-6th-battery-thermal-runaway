#!/usr/bin/env bash
# 전체 실행: 피처 캐시 -> 학습/제출 파일 -> 노트북.   실시간 확인: tail -f logs/latest.log
# 백그라운드:  nohup bash run_all.sh > logs/run_all.out 2>&1 &
cd "$(dirname "$0")"; source ./env.sh
set -o pipefail
bash run_features.sh  || { echo "features 실패"; exit 1; }
bash run_train.sh "$@" || { echo "train 실패"; exit 1; }
bash run_notebook.sh  || { echo "notebook 실패(제출 파일은 이미 생성됨)"; exit 1; }
echo "DONE. 제출 파일: $(pwd)/outputs/submission.csv"
