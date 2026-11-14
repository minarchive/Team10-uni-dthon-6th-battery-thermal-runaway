#!/usr/bin/env bash
# 융합 실험 계획을 우선순위대로 한 번에 실행. 센서 OOF 가 끝날 때까지 기다린 뒤 순서대로 진행하고, 단계가 실패해도 다음으로 넘어간다.
# 실행(로그가 터미널에 바로 찍힘):  bash scripts/run_fusion_plan.sh
#   NJOBS=8 bash scripts/run_fusion_plan.sh     CPU 공유 중이면 스레드 제한(기본 8)
# 후보마다 outputs/submissions/<이름>.csv 와 outputs/logs/candidates.csv 에 (Validation F1, 이미지가 바꾸는 행 수)를 기록.
# Validation F1 >= 0.85 인 후보는 로그에 '*** 보고 대상 ***' 으로 표시된다.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
NJOBS=${NJOBS:-8}
IMG=${IMG:-ft_ord_dinov2_b}
mkdir -p outputs/logs outputs/submissions
LOG=outputs/logs/fusion_plan_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "[$(date +%H:%M:%S)] ===== $* ====="; }
try()  { local tag=$1; shift; echo "[$(date +%H:%M:%S)] START $tag"; if "$@"; then echo "[$(date +%H:%M:%S)] DONE  $tag"; else echo "[$(date +%H:%M:%S)] FAIL  $tag (계속 진행)"; FAILED+=("$tag"); fi; }
FAILED=()
echo "log -> $LOG | img=$IMG njobs=$NJOBS"

step "0. 센서 OOF 대기 (sensor_oof.py)"
while [ ! -f outputs/preds/sensor_lgb_oof.npy ]; do
  echo "[$(date +%H:%M:%S)] 대기 중... 완료 fold: $(ls outputs/preds 2>/dev/null | grep -c '^_sensor_fold')/5"; sleep 60
done
tail -2 /tmp/sensor_oof.out 2>/dev/null

step "1. 스태커 (센서·이미지 확률 2단계 모델) -> gate_stack.csv"
try gate_stack python -u scripts/gate_stack.py --img "$IMG" --n_jobs "$NJOBS" --final --name gate_stack

step "2. 누적 임계값 분해 (임계값별 참조 모델) -> cum_route.csv"
try cum_route python -u scripts/cum_route.py --img "$IMG" --n_jobs "$NJOBS" --final --name cum_route

step "3. 프리즈 임베딩 + 센서 확률 조인트 헤드 -> joint_head.csv"
try joint_head python -u scripts/joint_head.py --img "$IMG" --n_jobs "$NJOBS" --final --name joint_head

step "4. 센서 오류 프레임 가중 이미지 파인튜닝 (공식 분할 1회 스크리닝, GPU)"
while pgrep -f "scripts/finetune.py" > /dev/null; do echo "[$(date +%H:%M:%S)] 실행 중인 파인튜닝 종료 대기..."; sleep 30; done
try ft_offbase python scripts/finetune.py --enc dinov2_b --name offbase --official 1 --bnd_gain 4 > outputs/logs/ft_offbase.log 2>&1
try ft_offsw   python scripts/finetune.py --enc dinov2_b --name offsw   --official 1 --bnd_gain 4 --sens_gain 3 > outputs/logs/ft_offsw.log 2>&1
python - <<'EOF' || echo "비교 스크립트 실패"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import ROOT, table
P = ROOT / "outputs" / "preds"
df = table(); lab = (df.label.values >= 1) & (df.fold.values >= 0)
y = df.label.values[lab] - 1; val = (df.split.values[lab] == "validation")
so = np.load(P / "sensor_lgb_oof.npy"); weak = (so.argmax(1) != y) | (so.max(1) < 0.7)
print("\n== 실험 6: 공식 분할 Validation 에서 이미지 단독 성능 (센서 약점 프레임 = 센서 오답 또는 확신<0.7)")
print(f"Validation 행 {val.sum()}, 그중 약점 프레임 {(weak & val).sum()}")
for nm in ["offbase", "offsw"]:
    try: p = np.load(P / f"ft_{nm}_oof.npy").argmax(1)
    except Exception as e: print(nm, "없음", e); continue
    f = lambda m: f1_score(y[m], p[m], average="macro", labels=list(range(6)), zero_division=0)
    print(f"{nm:8s} 전체 F1 {f(val):.4f} | 약점 프레임 정확도 {(p[val & weak] == y[val & weak]).mean():.4f} | 비약점 정확도 {(p[val & ~weak] == y[val & ~weak]).mean():.4f}")
EOF

step "5. 요약: 후보 CSV 와 이미지 영향도"
python - <<'EOF'
import pandas as pd
d = pd.read_csv("outputs/logs/candidates.csv")
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70)
print(d.drop(columns=["note"]).sort_values("val_f1_cap500", ascending=False).to_string(index=False))
hi = d[d.val_f1_cap500 >= 0.85]
print("\n*** Validation >= 0.85 후보:", ", ".join(hi.name + f"({v})" for name, v in zip(hi.name, hi.val_f1_cap500)) if len(hi) else "없음")
EOF
[ ${#FAILED[@]} -gt 0 ] && { echo "실패한 단계: ${FAILED[*]}"; } || echo "실패 없음"
echo "[$(date +%H:%M:%S)] ALL DONE (log: $LOG)"
