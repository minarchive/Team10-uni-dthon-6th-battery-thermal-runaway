"""atmp_t(주변온도, 글리치 777°C 포함) 처리 변형을 OOF 5-fold(fast)로 비교. 기준 = base + 가스합만.
A drop: atmp_t 를 상수로(모든 atmp_t 피처 제거 효과)   B first: atmp_t 를 실험의 첫 관측값(과거)으로 고정(주변온도 수준만 남기고 프레임별 글리치 제거)   C clip: [10,60] 클립"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; cls = fr.bat_class.values
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=8, verbose=-1, random_state=42)
d0 = raw.copy(); s = d0[GAS].sum(axis=1)
for g in GAS: d0[g] = 0.0
d0["meas_co2"] = s
order = np.lexsort((d0.relative_time_s.values, d0.experiment_id.values))
first = d0.iloc[order].groupby("experiment_id").atmp_t.transform("first"); first = first.reindex(d0.index)
print(f"atmp_t: 범위 {d0.atmp_t.min():.1f}~{d0.atmp_t.max():.1f}, 60°C 초과 행 {int((d0.atmp_t > 60).sum())} (train/val {int((d0.atmp_t[raw.label.values != -1] > 60).sum())}, test {int((d0.atmp_t[raw.label.values == -1] > 60).sum())})", flush=True)
base = None
for nm, mod in [("기준", None), ("A atmp_t 제거(상수)", "drop"), ("B atmp_t = 실험 첫 관측값", "first"), ("C atmp_t 클립[10,60]", "clip")]:
    d = d0.copy()
    if mod == "drop": d["atmp_t"] = 0.0
    elif mod == "first": d["atmp_t"] = first.values
    elif mod == "clip": d["atmp_t"] = d.atmp_t.clip(10, 60)
    X = F.build_features(d).iloc[lab_i].reset_index(drop=True); O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; m = lgb.LGBMClassifier(**P0).fit(X[~va], y[~va]); Z = np.zeros((va.sum(), 6)); Z[:, np.array(m.classes_) - 1] = m.predict_proba(X[va]); O[va] = Z
    p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)]); fv = f1(y, p)
    if base is None: base, bfs = fv, fs
    print(f"{nm:26s} OOF {fv:.4f} ({fv - base:+.4f}) 이긴 fold {int((fs > bfs).sum())}/5 | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
