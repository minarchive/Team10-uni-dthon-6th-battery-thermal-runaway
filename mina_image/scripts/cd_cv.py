"""C 가스 피처 제거(입력 가스열을 0 으로 만들어 피처가 상수가 되게 함, 가스합만 유지는 합을 meas_co2 에 넣음)  D (형태 x 단계) 균형 가중으로 파우치 소수 단계 보강. OOF 5-fold fast."""
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
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=16, verbose=-1, random_state=42)
def oof(X, w=None):
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; m = lgb.LGBMClassifier(**P0).fit(X[~va], y[~va], sample_weight=None if w is None else w[~va])
        pr = m.predict_proba(X[va]); O[va][:, :] = 0; Z = np.zeros((va.sum(), 6)); Z[:, np.array(m.classes_) - 1] = pr; O[va] = Z
    return O
base5 = np.load(ROOT / "outputs" / "preds" / "sensor_fast_oof.npy")
def rep(nm, O, b=None):
    p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)]); fv = f1(y, p)
    ex = "" if b is None else f" ({fv - b[0]:+.4f}) 이긴 fold {int((fs > b[1]).sum())}/5"
    print(f"{nm:34s} OOF {fv:.4f}{ex} | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True); return fv, fs
print("== C. 가스 피처 제거"); b = rep("가스 포함(기준, 5초 fast)", base5)
for nm, mode in [("가스 10종 제거(0 처리)", "drop"), ("가스합만 유지", "sum")]:
    d = raw.copy(); s = d[GAS].sum(axis=1)
    for g in GAS: d[g] = 0.0
    if mode == "sum": d["meas_co2"] = s
    X = F.build_features(d).iloc[lab_i].reset_index(drop=True); rep(nm, oof(X), b)
print("\n== D. (형태 x 단계) 균형 가중 (X1 피처, 기준 0.9079)")
X1 = pd.read_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet"); b = rep("X1 기준", oof(X1))
cell = cls * 10 + y; cnt = pd.Series(cell).map(pd.Series(cell).value_counts()).values.astype(float)
for pw in (0.25, 0.5, 1.0):
    w = cnt ** (-pw); w = w / w.mean(); rep(f"(형태x단계) 가중 지수 {pw}", oof(X1, w), b)
cls_only = pd.Series(cls).map(pd.Series(cls).value_counts()).values.astype(float)
for pw in (0.5,): w = cls_only ** (-pw); w = w / w.mean(); rep(f"형태만 균형 가중 지수 {pw}", oof(X1, w), b)
