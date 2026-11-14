"""L2/L3 전용 이미지 모델(finetune --bin23 1 의 OOF)을 센서와 결합해 L2/L3 경계만 보정할 수 있는지 OOF 로 평가.
대상 프레임: 센서(OOF, 후처리)가 L2 또는 L3 로 예측한 프레임. 이미지 점수 q = P(L3)/(P(L2)+P(L3)).
1) 구분력: 대상 프레임 중 실제 L2 vs L3 에 대한 AUC — 센서 P(>=L3) 대 전용 이미지 q 대 둘의 로지스틱 결합(실험 단위 fold CV)
2) 결합: 로지스틱(센서 logit, 이미지 logit; 파라미터 3개)을 학습 fold 로 적합해 검증 fold 에서 L2/L3 를 다시 정함 -> 후처리 F1 변화
임계값을 사람이 고르지 않고 로지스틱 확률 0.5 를 쓴다(선택 누수 최소)."""
import argparse, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from common import ROOT, table
ap = argparse.ArgumentParser(); ap.add_argument("--img", required=True); ap.add_argument("--tau", type=float, default=0.7); a = ap.parse_args()
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
so = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy"); io = np.load(ROOT / f"outputs/preds/ft_{a.img}_oof.npy")
ps = F.postprocess(fr, so, a.tau)
m = np.isin(ps, [2, 3]) & np.isin(y, [2, 3])           # 학습/AUC 평가용(실제 L2/L3)
tgt = np.isin(ps, [2, 3])                               # 보정 대상(센서가 L2/L3 로 본 프레임)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
s3 = so[:, 2:].sum(1); q = io[:, 2] / np.clip(io[:, 1] + io[:, 2], 1e-6, None)
Xs = np.stack([lg(s3), lg(q)], 1)
print(f"센서가 L2/L3 로 본 프레임 {tgt.sum()} (그중 실제 L2/L3 {m.sum()})")
print(f"AUC(L2 vs L3): 센서 P(>=L3) {roc_auc_score(y[m] == 3, s3[m]):.4f} | 전용 이미지 q {roc_auc_score(y[m] == 3, q[m]):.4f}")
new = ps.copy(); aucs = []
for k in range(5):
    tr, va = m & (fold != k), tgt & (fold == k)
    clf = LogisticRegression(C=1.0, max_iter=500).fit(Xs[tr], (y[tr] == 3).astype(int))
    pr = clf.predict_proba(Xs[va])[:, 1]; new[va] = np.where(pr >= 0.5, 3, 2)
    mv = m & (fold == k); aucs.append(roc_auc_score(y[mv] == 3, clf.predict_proba(Xs[mv])[:, 1]))
print(f"결합 로지스틱 AUC(fold 평균) {np.mean(aucs):.4f}")
hard = lambda p, idx=None: F.postprocess(fr, np.eye(6)[p - 1], 0.5)
b, n = hard(ps), hard(new)
f = lambda p, mk=slice(None): f1_score(y[mk], p[mk], labels=L, average="macro", zero_division=0)
fb = np.array([f(b, fold == k) for k in range(5)]); fn = np.array([f(n, fold == k) for k in range(5)])
print(f"OOF F1 (같은 단조 후처리): 센서 {f(b):.4f} -> 센서+전용이미지 {f(n):.4f} ({f(n) - f(b):+.4f}) | 이긴 fold {int((fn > fb).sum())}/5 | fold차이 {np.round(fn - fb, 4).tolist()}")
print(f"바뀐 프레임 {int((new != ps).sum())} 중 정답으로 바뀐 {int(((new != ps) & (new == y)).sum())}, 오답으로 바뀐 {int(((new != ps) & (ps == y)).sum())}")
