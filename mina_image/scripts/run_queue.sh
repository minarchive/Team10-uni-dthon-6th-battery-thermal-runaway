#!/usr/bin/env bash
# 무인 실험 큐: 앞 작업(run_fusion_plan.sh)이 끝나면 다음 실험으로 자동 진행하고 마지막에 요약 파일을 만든다.
#   1) 레짐 예측 가능성 probe (CPU, 몇 분)         -> outputs/logs/queue_regime_probe.txt
#   2) 시간 입력 이미지 모델: 60초 결과 비교 + 30초/120초 변형(공식 분할, GPU 각 약 10분)
#   3) 가장 좋은 시간 입력 변형이 기준(offbase)보다 이미지 단독 F1 +0.02 이상이면 5-fold 로 확정(OOF/test 확률 저장, 약 40분)
#   4) 요약                                       -> outputs/logs/queue_summary.txt
# 어떤 단계가 실패해도 다음으로 넘어간다. 데이터 파생물은 커밋하지 않는다(이 스크립트는 git 에 아무것도 하지 않는다).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
LOG=outputs/logs/queue_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "[$(date +%H:%M:%S)] ===== $* ====="; }
waitgpu() { while pgrep -f "scripts/finetune.py" > /dev/null; do sleep 20; done; }
echo "log -> $LOG"

step "1. 레짐 예측 가능성 probe"
python -u scripts/regime_probe.py 2>&1 | tee outputs/logs/queue_regime_probe.txt || echo "regime_probe 실패"

step "2. 앞 작업(run_fusion_plan.sh) 종료 대기"
while pgrep -f "run_fusion_plan.sh" > /dev/null; do echo "[$(date +%H:%M:%S)] 대기..."; sleep 60; done
waitgpu

step "3. 시간 입력 변형 (공식 분할): prev 30초, 120초"
for ps in 30 120; do
  [ -f outputs/preds/ft_offtmp${ps}_test.npy ] || { echo "[$(date +%H:%M:%S)] START offtmp$ps"; python scripts/finetune.py --enc dinov2_b --name offtmp$ps --official 1 --prev_sec $ps --bnd_gain 4 > outputs/logs/ft_offtmp$ps.log 2>&1 && echo DONE || echo FAIL; }
done

step "4. 비교와 승자 선택"
python - <<'EOF' | tee outputs/logs/queue_tmp_compare.txt
import json, numpy as np
from sklearn.metrics import f1_score
from common import ROOT, table
P = ROOT / "outputs" / "preds"
df = table(); lab = (df.label.values >= 1) & (df.fold.values >= 0)
y = df.label.values[lab] - 1; val = (df.split.values[lab] == "validation")
res = {}
for nm in ["offbase", "offtmp30", "offtmp60", "offtmp120"]:
    try:
        p = np.load(P / f"ft_{nm}_oof.npy").argmax(1)
    except Exception:
        continue
    pc = f1_score(y[val], p[val], average=None, labels=list(range(6)), zero_division=0)
    res[nm] = (f1_score(y[val], p[val], average="macro", labels=list(range(6)), zero_division=0), pc)
    print(f"{nm:10s} Validation 이미지 단독 Macro F1 {res[nm][0]:.4f} | 단계별 {np.round(pc, 3).tolist()}")
base = res.get("offbase", (None,))[0]
tmp = {k: v[0] for k, v in res.items() if k.startswith("offtmp")}
if base is not None and tmp:
    best = max(tmp, key=tmp.get)
    print(f"\n최고 시간 입력 {best}: {tmp[best]:.4f} vs 기준 offbase {base:.4f} ({tmp[best] - base:+.4f})")
    if tmp[best] - base >= 0.02:
        open("/tmp/queue_winner.txt", "w").write(best.replace("offtmp", ""))
        print("-> 기준 +0.02 이상: 5-fold 확정 실행")
    else:
        print("-> 기준 +0.02 미만: 5-fold 생략(잡음 범위)")
EOF
if [ -f /tmp/queue_winner.txt ]; then
  PS=$(cat /tmp/queue_winner.txt); rm -f /tmp/queue_winner.txt
  step "5. 5-fold 확정: ft_tmp${PS} (prev ${PS}초)"
  waitgpu
  python scripts/finetune.py --enc dinov2_b --name tmp${PS} --prev_sec $PS --bnd_gain 4 > outputs/logs/ft_tmp${PS}.log 2>&1 && echo DONE || echo FAIL
fi

step "6. 요약"
{ echo "== 후보 CSV와 이미지 영향도 (candidates.csv)"; cat outputs/logs/candidates.csv 2>/dev/null
  echo; echo "== 레짐 probe"; cat outputs/logs/queue_regime_probe.txt 2>/dev/null
  echo; echo "== 시간 입력 비교"; cat outputs/logs/queue_tmp_compare.txt 2>/dev/null
  echo; echo "== ft_results (최근)"; tail -4 outputs/logs/ft_results.csv | cut -c1-260
} > outputs/logs/queue_summary.txt
cat outputs/logs/queue_summary.txt
echo "[$(date +%H:%M:%S)] QUEUE DONE (log: $LOG)"
