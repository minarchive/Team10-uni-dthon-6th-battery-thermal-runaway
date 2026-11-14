#!/usr/bin/env bash
# 이미지 통계 피처 캐시 생성 (cache/img_feats_*.parquet). 이미 있으면 건너뜀. 재생성: bash run_features.sh --force
cd "$(dirname "$0")"; source ./env.sh
echo "DATA_DIR=$DATA_DIR  N_JOBS=$N_JOBS  python=$(python --version 2>&1)"
run_logged features python -u src/extract_image_feats.py "$@"
