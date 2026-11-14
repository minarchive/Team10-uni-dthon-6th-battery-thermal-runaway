"""LightGBM 무작위 탐색(OOF 5-fold, X1 피처 캐시). 과적합 점검: (1) 앞 3개 fold 로 고른 최적 설정이 뒤 2개 fold 에서 기준보다 나은지(중첩 확인) (2) 상위 설정을 3시드로 재평가."""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
X = pd.read_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet")
B = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=16, verbose=-1)
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
def run(ch, seed=42):
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; O[va] = lgb.LGBMClassifier(**{**B, **ch, "random_state": seed}).fit(X[~va], y[~va]).predict_proba(X[va])
    p = F.postprocess(fr, O, 0.7); return f1(y, p), np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
rng = np.random.default_rng(1)
space = dict(num_leaves=[7, 15, 23, 31, 47], min_child_samples=[20, 35, 50, 80, 120], colsample_bytree=[0.1, 0.15, 0.2, 0.3, 0.4], subsample=[0.6, 0.7, 0.8, 0.9],
             reg_lambda=[0.0, 1.0, 3.0, 10.0], reg_alpha=[0.0, 0.1, 1.0], min_split_gain=[0.0, 0.01, 0.1], max_bin=[63, 127, 255], path_smooth=[0.0, 1.0, 5.0])
sched = [dict(learning_rate=0.06, n_estimators=250), dict(learning_rate=0.1, n_estimators=150), dict(learning_rate=0.03, n_estimators=500)]
bf, bfs = run({}); print(f"기준 OOF {bf:.4f} fold {np.round(bfs, 4).tolist()}", flush=True)
res = []
for i in range(24):
    ch = {k: v[rng.integers(len(v))] for k, v in space.items()}; ch.update(sched[rng.integers(3)])
    t0 = time.time(); f, fs = run(ch); res.append((f, fs, ch))
    print(f"[{i + 1:2d}] OOF {f:.4f} ({f - bf:+.4f}) fold {np.round(fs, 3).tolist()} | {({k: v for k, v in ch.items()})} | {time.time() - t0:.0f}s", flush=True)
res.sort(key=lambda r: -r[0])
print("\n상위 5:"); [print(f"  {r[0]:.4f} ({r[0] - bf:+.4f}) {r[2]}") for r in res[:5]]
# (1) 중첩 확인: 앞 3개 fold(0,1,2)에서 고른 최적이 뒤 2개 fold(3,4)에서 기준보다 나은가
sel = max(res, key=lambda r: r[1][:3].mean()); print(f"\n[중첩] fold0~2 평균으로 고른 설정: fold0~2 {sel[1][:3].mean():.4f} (기준 {bfs[:3].mean():.4f}) -> 보지 않은 fold3~4 {sel[1][3:].mean():.4f} (기준 {bfs[3:].mean():.4f}, 차이 {sel[1][3:].mean() - bfs[3:].mean():+.4f})")
sel2 = max(res, key=lambda r: r[1][2:].mean()); print(f"[중첩] fold2~4 로 고른 설정의 보지 않은 fold0~1: {sel2[1][:2].mean():.4f} (기준 {bfs[:2].mean():.4f}, 차이 {sel2[1][:2].mean() - bfs[:2].mean():+.4f})")
# (2) 상위 3 설정 3시드 재평가
b3 = np.mean([run({}, s)[0] for s in (42, 7, 123)])
for r in res[:3]:
    s3 = [run(r[2], s)[0] for s in (42, 7, 123)]; print(f"[3시드] {np.mean(s3):.4f} (기준 3시드 {b3:.4f}, {np.mean(s3) - b3:+.4f}) {r[2]}", flush=True)
