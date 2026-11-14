#!/usr/bin/env bash
# EDA v3 (CPU 만, 모델 재학습 없음) 노트북 생성 + 실행 (outputs/oof_gate.parquet 필요). 결과: eda_v3_executed.ipynb(+html)
cd "$(dirname "$0")"; source ./env.sh
run_logged eda_v3 bash -c '
  python -u src/make_eda_v3.py &&
  jupyter nbconvert --to notebook --execute eda_v3.ipynb --output eda_v3_executed.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 &&
  jupyter nbconvert --to html eda_v3_executed.ipynb --output eda_v3.html'
