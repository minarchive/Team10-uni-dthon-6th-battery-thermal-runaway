#!/usr/bin/env bash
# ablation -> 설정/후처리 선택 -> train+val 재학습 -> outputs/submission.csv
# 옵션은 run_train.py 에 그대로 전달: bash run_train.sh --smoke | --no_cv | --n_aug 3 ...
cd "$(dirname "$0")"; source ./env.sh
run_logged train python -u src/run_train.py "$@"
