#!/usr/bin/env bash
# 파인튜닝 1차 패널(전부 같은 5-fold, 같은 하이퍼). 로그는 outputs/logs/ft_<name>.log
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p outputs/logs
for enc in dino_r50 mae_b dinov2_b resnet50; do
  python scripts/finetune.py --enc "$enc" --name "$enc" 2>&1 | tee "outputs/logs/ft_${enc}.log"
done
