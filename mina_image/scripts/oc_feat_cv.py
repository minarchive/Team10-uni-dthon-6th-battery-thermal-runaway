"""과충전 L3 신호(thermal-atmp, T2-T1 등)를 LightGBM 컬럼으로 추가했을 때의 효과 + 규칙 결합. 기준 = X1 + 가스합만(OOF 0.912). 모두 인과적.
F1 컬럼: thermal-amb 의 누적최댓값·임계 도달 후 경과, T2-T1 의 누적최솟값·임계 이하 도달 후 경과, thermal_t 30초 기울기 누적최댓값
F2 방식 분리: F1 의 각 컬럼을 과충전일 때만 값(가열은 NaN)으로 만든 복사본도 추가
H  규칙 결합: 과충전 행에서 모델이 L2/L3 로 예측하면 thermal-amb 누적최댓값 >= theta(학습 fold 과충전 실험에서 정함)로 L2/L3 결정"""
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
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; cls = fr.bat_class.values; oc = (fr.tr_method.values == 2)
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=8, verbose=-1, random_state=42)
d = raw.copy(); s = d[GAS].sum(axis=1)
for g in GAS: d[g] = 0.0
d["meas_co2"] = s
base = F.build_features(d)
eid = d.experiment_id.values; t_all = d.relative_time_s.values.astype(float); idxs = pd.Series(np.arange(len(d))).groupby(eid).indices
def cumx(x, kind):
    o = np.full(len(d), np.nan)
    for e, ix in idxs.items():
        ix = ix[np.argsort(t_all[ix], kind="stable")]; xs = pd.Series(x[ix]); o[ix] = (xs.cummax() if kind == "max" else xs.cummin()).values
    return o
def since(x, thr, up=True):
    o = np.full(len(d), np.nan)
    for e, ix in idxs.items():
        ix = ix[np.argsort(t_all[ix], kind="stable")]; t = t_all[ix]; m = (x[ix] >= thr) if up else (x[ix] <= thr); m = np.maximum.accumulate(m)
        tf = t[np.argmax(m)] if m.any() else np.inf; o[ix] = np.where(t >= tf, t - tf, np.nan)
    return o
tha = (d.thermal_t - d.atmp_t).values; t21 = (d.meas_t2 - d.meas_t1).values
sl = np.full(len(d), np.nan); x = d.thermal_t.values.astype(float)
for e, ix in idxs.items():
    ix = ix[np.argsort(t_all[ix], kind="stable")]; tt = t_all[ix]; j = np.searchsorted(tt, tt - 30, side="right") - 1; ok = j >= 0; o = np.full(len(ix), np.nan); o[ok] = (x[ix][ok] - x[ix][j[ok]]) / np.maximum(tt[ok] - tt[j[ok]], 1); sl[ix] = pd.Series(o).cummax().values
F1 = {"tha_cummax": cumx(tha, "max"), "t21_cummin": cumx(t21, "min"), "thslope_cummax": sl}
for th in (2, 4, 6, 8, 12, 16, 24): F1[f"tha_since{th}"] = since(tha, th, True)
for th in (0, -2, -5, -10): F1[f"t21_since_le{-th if th <= 0 else th}"] = since(t21, th, False)
F1 = pd.DataFrame(F1, index=base.index)   # base(index=sample_id)와 정렬되도록
ocall = (d.tr_method.values == 2)
F2 = F1.mask(~pd.Series(ocall, index=F1.index), axis=0).add_suffix("_oc")   # 가열 행은 NaN
tha_cm = F1["tha_cummax"].values
def sub(*parts): return pd.concat([base] + list(parts), axis=1).iloc[lab_i].reset_index(drop=True)
def run(X, seed=42):
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; m = lgb.LGBMClassifier(**{**P0, "random_state": seed}).fit(X[~va], y[~va]); Z = np.zeros((va.sum(), 6)); Z[:, np.array(m.classes_) - 1] = m.predict_proba(X[va]); O[va] = Z
    return O
def rep(nm, O, b=None):
    p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)]); fv = f1(y, p)
    ex = "" if b is None else f" ({fv - b[0]:+.4f}) 이긴 fold {int((fs > b[1]).sum())}/5"
    print(f"{nm:30s} OOF {fv:.4f}{ex} | 과충전 {f1(y[oc], p[oc]):.4f} 가열 {f1(y[~oc], p[~oc]):.4f} | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True); return fv, fs, p
O0 = run(sub()); b = rep("기준(X1 없이 base+가스합)", O0); np.save(ROOT / "outputs/preds/oof_gassum.npy", O0.astype(np.float32))
O1 = run(sub(F1)); rep("+ 과충전 L3 컬럼(F1)", O1, b[:2])
O2 = run(sub(F1, F2)); rep("+ F1 + 방식 분리 복사본(F2)", O2, b[:2]); np.save(ROOT / "outputs/preds/oof_gassum_oc2.npy", O2.astype(np.float32))
# H: 규칙 결합(OOF fold 안에서 theta 결정)
for nm, Ob in [("기준 + 규칙 override", O0), ("F2 모델 + 규칙 override", O2)]:
    Oh = Ob.copy(); tha_l = tha_cm[lab_i]
    for k in range(5):
        va = fold == k; trm = (~va) & oc & np.isin(y, [2, 3]); grid = np.unique(np.nanpercentile(tha_l[trm], np.linspace(2, 98, 33))); bt, bf = grid[0], -1
        for th in grid:
            f = f1_score((y[trm] == 3).astype(int), (tha_l[trm] >= th).astype(int), zero_division=0)
            if f > bf: bf, bt = f, th
        rows = va & oc
        P = Oh[rows]; p23 = P[:, 1] + P[:, 2]; l3 = tha_l[rows] >= bt
        sel = (P.argmax(1) <= 2) & (P.argmax(1) >= 1)                    # 모델이 L2/L3 로 본 과충전 행
        newP = P.copy(); newP[sel, 1] = np.where(l3[sel], 0.0, p23[sel]); newP[sel, 2] = np.where(l3[sel], p23[sel], 0.0); Oh[rows] = newP
    rep(nm, Oh, b[:2]); np.save(ROOT / "outputs/preds" / ("oof_gassum_override.npy" if nm.startswith("기준") else "oof_gassum_oc2_override.npy"), Oh.astype(np.float32))
