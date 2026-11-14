#!/usr/bin/env bash
# 센서 LightGBM+XGBoost 앙상블 + 이미지 백본(ft_dinov2_b) 융합 -> Validation 점수 -> 제출 CSV (한 번에).
# 실행(로그가 터미널에 바로 찍히고 파일에도 저장):
#   bash scripts/run_ens_submit.sh                 # 기본: ft_dinov2_b, 전체
#   SMOKE=1 bash scripts/run_ens_submit.sh         # 1~2분 배선 확인
#   NJOBS=8 bash scripts/run_ens_submit.sh         # CPU 공유 중일 때 스레드 제한
#   FAST=1  bash scripts/run_ens_submit.sh         # 트리 수 절반
#   BACKBONE=ft_dinov2_b,ft_mae_b NAME=ens_v2 bash scripts/run_ens_submit.sh   # 백본 평균(각 *_test.npy 필요)
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p outputs/logs outputs/preds outputs/submissions
NAME=${NAME:-ens_lgb_xgb}
BACKBONE=${BACKBONE:-ft_dinov2_b}          # 이미 실행한 파인튜닝 중 val 최고(F1 0.781) + test 예측 보유
LOG=outputs/logs/${NAME}_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "[$(date +%H:%M:%S)] ===== $* ====="; }
die()  { echo "[$(date +%H:%M:%S)] FAIL: $*"; exit 1; }
echo "log -> $LOG | name=$NAME backbone=$BACKBONE smoke=${SMOKE:-0} fast=${FAST:-0}"

step "0. 환경 (xgboost/lightgbm)"
python - <<'EOF' || pip install -q xgboost
import xgboost, lightgbm; print("xgboost", xgboost.__version__, "lightgbm", lightgbm.__version__)
EOF
python -c "import xgboost,lightgbm;print('OK', xgboost.__version__)" || die "xgboost 설치 실패"

step "1. 선행 산출물 확인/생성"
[ -f data/folds.parquet ] || python scripts/make_folds.py || die folds
IFS=',' read -ra BB <<< "$BACKBONE"
NEED=0
for b in "${BB[@]}"; do
  [ -f "outputs/preds/${b}_oof.npy" ] || die "${b}_oof.npy 없음 (finetune.py 먼저)"
  [ -f "outputs/preds/${b}_test.npy" ] || die "${b}_test.npy 없음: python scripts/finetune.py --enc ${b#ft_} --name ${b#ft_} 로 test 예측 생성 필요"
done
python scripts/build_image_features.py --models "${BB[@]}" || die build_image_features

step "2. 앙상블 학습 -> Validation(Train->Val) 보고 -> Train+Val 재학습 -> 제출 CSV"
ARGS=(--models "${BB[@]}" --name "$NAME")
[ "${SMOKE:-0}" = 1 ] && ARGS+=(--smoke)
[ "${FAST:-0}" = 1 ] && ARGS+=(--fast)
[ -n "${NJOBS:-}" ] && ARGS+=(--n_jobs "$NJOBS")   # CPU 가 바쁠 때(예: NJOBS=8)
python -u scripts/ens_submit.py "${ARGS[@]}" || die ens_submit

step "3. 결과"
SUF=""; [ "${SMOKE:-0}" = 1 ] && SUF=_smoke
cat "outputs/submissions/${NAME}${SUF}.json"
head -3 "outputs/submissions/${NAME}${SUF}.csv"; wc -l "outputs/submissions/${NAME}${SUF}.csv"
echo "제출 파일: $(pwd)/outputs/submissions/${NAME}${SUF}.csv"
echo "[$(date +%H:%M:%S)] ALL DONE (log: $LOG)"
