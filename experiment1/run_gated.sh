#!/usr/bin/env bash
# 센서 주도 + 이미지 게이팅 실험 (run_train.sh 가 만든 outputs/summary.json 의 센서 설정을 사용).  --smoke 지원
cd "$(dirname "$0")"; source ./env.sh
run_logged gated python -u src/run_gated.py "$@"
