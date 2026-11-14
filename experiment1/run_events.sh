#!/usr/bin/env bash
# 전처리/사건 피처 효과 검증 (A: 기존, B: +사건 피처, C: 인과 클리닝+사건 피처). 실험 단위 5-fold OOF 비교
cd "$(dirname "$0")"; source ./env.sh
run_logged events python -u src/run_events_cv.py "$@"
