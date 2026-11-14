#!/usr/bin/env bash
# 스케일링/이미지 가공 증거 실험 (결과: outputs/evidence/*.csv,*.png).  --smoke 지원
cd "$(dirname "$0")"; source ./env.sh
run_logged evidence python -u src/evidence_scaling_image.py "$@"
