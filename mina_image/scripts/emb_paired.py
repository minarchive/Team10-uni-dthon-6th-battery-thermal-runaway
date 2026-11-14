"""임베딩 PCA 피처 이득이 진짜인지: 시드 3개 평균 OOF 로 센서 단독과 실험 단위 짝지은 비교(실험별 ΔF1 부호·평균, 부트스트랩 신뢰구간)."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
from fusion_lib import emb_columns
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7)); BASE0 = list(F.BASE_SIGNALS)
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; cls = fr.bat_class.values
ids = tb.sample_id.astype(str).values; eids = fr.experiment_id.values; ue = np.unique(eids); idx = {e: np.where(eids == e)[0] for e in ue}
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
def oof(k, seed):
    d = raw.copy(); extra = []
    if k:
        E = emb_columns(ids, lab, "dinov2_b:b08_mean", k, seed); extra = list(E.columns)
        for c in extra: d[c] = E[c].values
    F.BASE_SIGNALS = BASE0 + extra; F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    X = F.build_features(d).iloc[lab_i].reset_index(drop=True); O = np.zeros((len(y), 6))
    for kf in range(5):
        va = fold == kf; m = F.build_model(); m.set_params(n_jobs=16, n_estimators=250, learning_rate=0.06, random_state=42 + seed)
        O[va] = m.fit(X[~va], y[~va]).predict_proba(X[va])
    return O
res = {}
for k in (0, 4, 8):
    res[k] = np.mean([oof(k, s) for s in (0, 1, 2)], 0); p = F.postprocess(fr, res[k], 0.7)
    print(f"PCA{k}: 3시드 평균 OOF {f1(y, p):.4f} | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]}", flush=True)
pb = F.postprocess(fr, res[0], 0.7); pe0 = np.array([f1(y[idx[e]], pb[idx[e]]) for e in ue]); ec = np.array([cls[idx[e][0]] for e in ue])
rng = np.random.default_rng(0)
for k in (4, 8):
    pk = F.postprocess(fr, res[k], 0.7); d = np.array([f1(y[idx[e]], pk[idx[e]]) for e in ue]) - pe0
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"\nPCA{k} 대비 센서 단독, 실험 단위 ΔF1: 평균 {d.mean():+.4f} (95% 신뢰구간 {np.percentile(bs, 2.5):+.4f}~{np.percentile(bs, 97.5):+.4f}) | 개선 {int((d > 0.001).sum())}개, 악화 {int((d < -0.001).sum())}개")
    for c in (1, 2, 3):
        dc = d[ec == c]; bc = [dc[rng.integers(0, len(dc), len(dc))].mean() for _ in range(2000)]
        print(f"   bat_class {c}: 실험 {len(dc)} 평균 {dc.mean():+.4f} (CI {np.percentile(bc, 2.5):+.4f}~{np.percentile(bc, 97.5):+.4f}) 개선 {int((dc > 0.001).sum())}/악화 {int((dc < -0.001).sum())}")
