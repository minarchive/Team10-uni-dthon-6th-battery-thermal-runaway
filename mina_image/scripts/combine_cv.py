"""튜닝 이득 후보 조합 + 시드 앙상블 + 후처리 τ 조정 (OOF 5-fold, X1 피처 캐시 사용). 선택은 OOF 로만."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import ROOT, table, DATA
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); fr["relative_time_s"] = fr.relative_time_s.astype(float)
y = fr.label.astype(int).values; fold = fr.fold.values
X = pd.read_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet")
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=16, verbose=-1)
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
def oof(ch, seeds):
    O = np.zeros((len(y), 6))
    for sd in seeds:
        for k in range(5):
            va = fold == k
            O[va] += lgb.LGBMClassifier(**{**P0, **ch, "random_state": sd}).fit(X[~va], y[~va]).predict_proba(X[va])
    return O / len(seeds)
base = np.load(ROOT / "outputs/preds/tune_C0_oof.npy"); bp = F.postprocess(fr, base, 0.7)
bfs = np.array([f1(y[fold == k], bp[fold == k]) for k in range(5)])
res = {}
for nm, ch, seeds in [("C3 colsample0.15 (시드1)", dict(colsample_bytree=0.15), [42]), ("C3+C9 colsample0.15+extra_trees (시드1)", dict(colsample_bytree=0.15, extra_trees=True), [42]),
                      ("C3+C9 3시드 평균", dict(colsample_bytree=0.15, extra_trees=True), [42, 7, 123]), ("C0 3시드 평균(기준 설정)", {}, [42, 7, 123])]:
    O = oof(ch, seeds); res[nm] = O
    p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    print(f"{nm:40s} OOF {f1(y, p):.4f} (C0 시드1 대비 {f1(y, p) - f1(y, bp):+.4f}) 이긴 fold {int((fs > bfs).sum())}/5", flush=True)
best = max(res, key=lambda k: f1(y, F.postprocess(fr, res[k], 0.7)))
print(f"\n최고 OOF 설정: {best}\n후처리 τ 스윕 (해당 설정):")
for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]:
    p = F.postprocess(fr, res[best], tau); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    print(f"  τ={tau:<4} OOF {f1(y, p):.4f} | fold {np.round(fs, 4).tolist()}", flush=True)
np.save(ROOT / "outputs" / "preds" / "combine_best_oof.npy", res[best].astype(np.float32))
