#!/usr/bin/env bash
# 순서형 누적 헤드 + 전환 경계 샘플링 파인튜닝 패널(순차, test 예측 있으면 건너뜀). 실행: bash scripts/run_ft_ordinal.sh
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1
mkdir -p outputs/logs
ft() { local name=$1 enc=$2; shift 2
  [ -f "outputs/preds/ft_${name}_test.npy" ] && { echo "skip $name"; return; }
  echo "[$(date +%H:%M:%S)] START $name ($enc $*)"
  python scripts/finetune.py --enc "$enc" --name "$name" "$@" > "outputs/logs/ft_${name}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE  $name: $(tail -1 outputs/logs/ft_${name}.log | cut -c1-230)" || echo "[$(date +%H:%M:%S)] FAIL  $name"; }
ft ord_dinov2_b  dinov2_b --ordinal 1 --bnd_gain 4      # 순서형 + 경계
ft ord_resnet50  resnet50 --ordinal 1 --bnd_gain 4
ft ordonly_dinov2_b dinov2_b --ordinal 1                # 분리 ablation
ft bndonly_dinov2_b dinov2_b --bnd_gain 4
echo "[$(date +%H:%M:%S)] ALL FT DONE"
