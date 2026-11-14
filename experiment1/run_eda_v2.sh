#!/usr/bin/env bash
# EDA v2 노트북 생성 + 실행 (outputs/oof_gate.parquet 필요: run_gated.sh 먼저). 결과: eda_v2_executed.ipynb(+html)
cd "$(dirname "$0")"; source ./env.sh
run_logged eda_v2 bash -c '
  python -u src/make_eda_v2.py &&
  jupyter nbconvert --to notebook --execute eda_v2.ipynb --output eda_v2_executed.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 &&
  jupyter nbconvert --to html eda_v2_executed.ipynb --output eda_v2.html'
