"""가스 피처 제거 이득 확인. X1(base + thermal 임계)에서 가스를 정리한 변형을 OOF 5-fold fast 로 비교, 시드 2개 평균(잡음 구분용: 이번 한 번만).
변형: 기준(가스 포함) / 가스합만(개별 가스 0) / 가스 전부 제거 / 가스합 + CO2·CO 만(대표 2종) / 가스합만 + 정규화 없는 간단 확인."""
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
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=16, verbose=-1)
def cross_feats(df, spec):
    d = df.sort_values(["experiment_id", "relative_time_s"]); eid, t = d.experiment_id, d.relative_time_s; g = d.groupby("experiment_id"); out = {}
    for name, (fn, levels) in spec.items():
        cm = fn(d, g).groupby(eid).cummax()
        for lv in levels: out[f"cx_{name}_{lv}"] = t - t.where(cm >= lv).groupby(eid).cummin()
    return pd.DataFrame(out).loc[df.index]
SPEC1 = {"thermal": (lambda d, g: d.thermal_t, [35, 40, 45, 50, 55, 60, 65, 70, 80, 100, 150, 200]), "thermal_rel": (lambda d, g: d.thermal_t - g.thermal_t.transform("first"), [2, 5, 10, 15, 20, 30, 40])}
def build(mode):
    d = raw.copy(); s = d[GAS].sum(axis=1)
    if mode != "base":
        keep = {"sum": [], "none": [], "co": ["meas_co2", "meas_co"]}[mode]
        for g in GAS:
            if g not in keep: d[g] = 0.0
        if mode in ("sum", "co") and "meas_co2" not in keep: d["meas_co2"] = s
        if mode == "co": pass
    X = F.build_features(d); X = pd.concat([X, cross_feats(d, SPEC1).set_index(X.index)], axis=1)
    return X.iloc[lab_i].reset_index(drop=True)
def run(X):
    outs = []
    for sd in (42, 7):
        O = np.zeros((len(y), 6))
        for k in range(5):
            va = fold == k; m = lgb.LGBMClassifier(**P0, random_state=sd).fit(X[~va], y[~va]); Z = np.zeros((va.sum(), 6)); Z[:, np.array(m.classes_) - 1] = m.predict_proba(X[va]); O[va] = Z
        outs.append(O)
    return np.mean(outs, 0)
base = None
for nm, mode in [("기준(가스 포함, X1)", "base"), ("가스합만", "sum"), ("가스 전부 제거", "none"), ("가스합 + CO2 + CO", "co")]:
    O = run(build(mode)); p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)]); fv = f1(y, p)
    if base is None: base, bfs = fv, fs
    print(f"{nm:22s} OOF(2시드 평균) {fv:.4f} ({fv - base:+.4f}) 이긴 fold {int((fs > bfs).sum())}/5 | fold {np.round(fs, 3).tolist()} | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
