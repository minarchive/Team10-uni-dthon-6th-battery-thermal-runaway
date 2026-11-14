#!/usr/bin/env bash
# 순차 실험 파이프라인 (중단 후 다시 실행해도 끝난 단계는 건너뜀). 실패한 단계는 outputs/logs/pipeline_failed.txt 에 기록하고 계속 진행.
# 실행: nohup bash scripts/run_pipeline.sh > outputs/logs/pipeline.log 2>&1 &
# 예상 소요(거친 추정): 시간 문맥 probe 수 분 + 파인튜닝 11회 x 약 10분 = 약 2시간
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p outputs/logs outputs/preds
FAIL=outputs/logs/pipeline_failed.txt

run() {  # run <표시이름> <명령...>
  local tag=$1; shift
  echo "[$(date +%H:%M:%S)] START $tag"
  if "$@"; then echo "[$(date +%H:%M:%S)] DONE  $tag"; else echo "$tag" >> "$FAIL"; echo "[$(date +%H:%M:%S)] FAIL  $tag"; fi
}

# 0) 공통 준비물
[ -f data/folds.parquet ] || run folds python scripts/make_folds.py
[ -f data/img224_pad.npy ] || run cache_images python scripts/cache_images.py

# 1) 시간 문맥 probe (가장 값싼 핵심 질문: 과거 프레임이 단일 프레임보다 나은가, 5/10/15초/혼합에서 유지되는가)
for el in dino_r50:s4_gap mae_b:b08_mean dinov2_b:b05_mean; do
  enc=${el%%:*}; layer=${el##*:}
  grep -q "^${enc},${layer}," outputs/logs/temporal_results.csv 2>/dev/null || run "temporal_${enc}_${layer}" python scripts/temporal_probe.py --enc "$enc" --layer "$layer"
done

# 2) frozen probe 의 test 예측 저장(센서 모델 전달용). 같은 시드라 OOF 점수는 동일
for el in dino_r50:s4_gap mae_b:b08_mean dinov2_b:b05_mean resnet50:s3_gap; do
  enc=${el%%:*}; layer=${el##*:}
  [ -f "outputs/preds/${enc}__${layer}_test.npy" ] || run "probe_test_${enc}_${layer}" python scripts/probe_cv.py --encoder "$enc" --layers "$layer" --force
done

# 3) 파인튜닝 패널 (test 예측 npy 가 있으면 건너뜀)
ft() {  # ft <이름> <인코더> [추가 인자...]
  local name=$1 enc=$2; shift 2
  [ -f "outputs/preds/ft_${name}_test.npy" ] || run "ft_${name}" bash -c "python scripts/finetune.py --enc $enc --name $name $* > outputs/logs/ft_${name}.log 2>&1"
}
ft dino_r50 dino_r50
ft mae_b mae_b
ft dinov2_b dinov2_b
ft resnet50 resnet50

# 4) dino_r50 ablation (한 번에 한 가지만 바꿈; 기준 = 기본 설정 dino_r50)
ft dino_r50_notau dino_r50 --tau 0            # 순서형 soft label 끄기(one-hot + label smoothing)
ft dino_r50_nobal dino_r50 --power 0          # 클래스 균형 샘플링 끄기
ft dino_r50_noaug dino_r50 --aug 0            # 증강 끄기
ft dino_r50_mlp dino_r50 --head mlp           # 헤드: Dropout + LeakyReLU MLP
ft dino_r50_cut3 dino_r50 --cut 3             # layer4 제외(stage3까지)
ft dino_r50_lr3e4 dino_r50 --lr 3e-4          # 백본 학습률 3배
ft dino_r50_ep20 dino_r50 --epochs 20 --patience 6

# 5) 시계열 모델 전달용 이미지 확률 테이블 + 앙상블/오류 겹침 분석 + 요약
M="dino_r50__s4_gap mae_b__b08_mean dinov2_b__b05_mean resnet50__s3_gap"
for t in ft_dino_r50 ft_mae_b ft_dinov2_b ft_resnet50; do [ -f "outputs/preds/${t}_test.npy" ] && M="$M $t"; done
run build_image_features python scripts/build_image_features.py --models $M
run summarize python scripts/summarize.py
run ensemble_oof bash -c "python scripts/ensemble_oof.py > outputs/logs/ensemble_report.txt 2>&1"

echo "[$(date +%H:%M:%S)] ALL STAGES FINISHED"
[ -f "$FAIL" ] && { echo "실패한 단계:"; cat "$FAIL"; } || echo "실패 없음"
