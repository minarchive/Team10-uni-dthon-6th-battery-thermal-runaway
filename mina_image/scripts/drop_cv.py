"""학습 데이터 일부 제거 가설 검증(OOF 5-fold fast, X1 피처). 평가는 항상 모든 프레임, 제거는 학습 fold 에서만.
D1: 학습 실험 중 센서 OOF F1 하위 q% 실험 제거(q=10,20)  *제거 기준은 학습 fold 실험의 기존 OOF 점수(검증 fold 실험의 정보는 쓰지 않음; 단 그 OOF 점수 자체가 검증 fold 실험을 학습에 썼던 모델의 것이라 아주 약한 누수가 있음 — 스크리닝용)
D2: 라벨 전환점 ±m 프레임 학습 제외(m=2,6)"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
X = pd.read_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet")
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=16, verbose=-1, random_state=42)
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
eids = fr.experiment_id.values; ue = np.unique(eids); idx = {e: np.where(eids == e)[0] for e in ue}
so = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy"); pb = F.postprocess(fr, so, 0.7)
pe = pd.Series({e: f1(y[idx[e]], pb[idx[e]]) for e in ue})
t = fr.relative_time_s.values.astype(float)
near = {}
for m in (2, 6):
    nm = np.zeros(len(y), bool)
    for e in ue:
        ix = idx[e][np.argsort(t[idx[e]], kind="stable")]; l = y[ix]; ch = np.where(np.diff(l) != 0)[0] + 1
        for c in ch: nm[ix[max(0, c - m):c + m]] = True
    near[m] = nm
cases = [("기준", np.ones(len(y), bool))]
for q in (10, 20):
    bad = set(pe.sort_values().index[:int(len(pe) * q / 100)]); cases.append((f"D1 하위{q}% 실험 제거(학습만)", ~np.isin(eids, list(bad))))
for m in (2, 6): cases.append((f"D2 경계 ±{m}프레임 제외(학습만)", ~near[m]))
base = None
for nm, keep in cases:
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; trm = (~va) & keep
        O[va] = lgb.LGBMClassifier(**P0).fit(X[trm], y[trm]).predict_proba(X[va])
    p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    if base is None: base, bfs = f1(y, p), fs
    print(f"{nm:28s} 학습 행 {int(keep.sum()):6d} | OOF {f1(y, p):.4f} ({f1(y, p) - base:+.4f}) 이긴 fold {int((fs > bfs).sum())}/5 | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
