#!/usr/bin/env bash
# 노트북 생성 + 실행 (EDA 시각화 + 모델 학습). 결과: experiment1_executed.ipynb (커밋 대상 아님: 데이터 파생물 포함)
cd "$(dirname "$0")"; source ./env.sh
run_logged notebook bash -c '
  python -u src/make_notebook.py &&
  jupyter nbconvert --to notebook --execute experiment1.ipynb --output experiment1_executed.ipynb \
     --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 --log-level=INFO'
