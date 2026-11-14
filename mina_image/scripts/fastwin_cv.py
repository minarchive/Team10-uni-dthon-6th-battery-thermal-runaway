"""급변 구간용 짧은 창 피처 검증(OOF 5-fold fast, X1 피처 + 신규). 모두 같은 실험의 현재·과거 관측만 사용, 시간(초) 기준.
신호별: 10초/15초 변화율, 가속도(10초 변화율의 10초 전 대비 차), 최근 30초 안 한 칸 변화율의 최대/최소.
지표: 합산 OOF, bat_class별, 라벨 점프 실험(L3->L5, L4->L6) 11개의 F1 과 점프 전후 ±12프레임 오류율."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
raw["gas_sum"] = np.log1p(raw[GAS].sum(axis=1)); raw["t_max"] = raw[["meas_t1", "meas_t2"]].max(axis=1)
SIG = ["meas_t1", "meas_t2", "thermal_t", "meas_p", "meas_v", "gas_sum", "t_max"]
eid = raw.experiment_id.values; t_all = raw.relative_time_s.values.astype(float)
out = {}
order_all = np.lexsort((t_all, eid))
for s in SIG:
    x_all = raw[s].values.astype(float)
    cols = {k: np.full(len(raw), np.nan) for k in ["d10", "d15", "acc10", "smax30", "smin30"]}
    for e, ix in pd.Series(np.arange(len(raw))).groupby(eid).indices.items():
        ix = ix[np.argsort(t_all[ix], kind="stable")]; t = t_all[ix]; x = x_all[ix]
        def lag(a, sec):
            j = np.searchsorted(t, t - sec, side="right") - 1; r = np.full(len(a), np.nan); ok = j >= 0; r[ok] = a[j[ok]]; return r
        d10 = (x - lag(x, 10)) / 10.0; d15 = (x - lag(x, 15)) / 15.0
        acc = d10 - lag(d10, 10)
        dt = np.r_[np.nan, np.diff(t)]; d1 = np.r_[np.nan, np.diff(x)] / dt
        ser = pd.Series(d1, index=pd.to_datetime(t, unit="s"))
        smax = ser.rolling("30s").max().values; smin = ser.rolling("30s").min().values
        for k, v in zip(["d10", "d15", "acc10", "smax30", "smin30"], [d10, d15, acc, smax, smin]): cols[k][ix] = v
    for k, v in cols.items(): out[f"fw_{s}_{k}"] = v
FW = pd.DataFrame(out)
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; cls = fr.bat_class.values
X1 = pd.read_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet")
FWl = FW.iloc[lab_i].reset_index(drop=True)
print(f"신규 피처 {FW.shape[1]}개 (결측 비율 {FW.isna().mean().mean():.3f})", flush=True)
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=16, verbose=-1, random_state=42)
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
# 라벨 점프 실험
t_l = fr.relative_time_s.values.astype(float); e_l = fr.experiment_id.values
jump_exp, near = set(), np.zeros(len(y), bool)
for e in np.unique(e_l):
    ix = np.where(e_l == e)[0]; ix = ix[np.argsort(t_l[ix], kind="stable")]; ch = np.where(np.diff(y[ix]) > 1)[0] + 1
    if len(ch): jump_exp.add(e)
    for c in ch: near[ix[max(0, c - 12):c + 12]] = True
isj = np.isin(e_l, list(jump_exp)); base = None
for nm, X in [("X1 기준", X1), ("X1 + 짧은 창 피처", pd.concat([X1, FWl], axis=1))]:
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; O[va] = lgb.LGBMClassifier(**P0).fit(X[~va], y[~va]).predict_proba(X[va])
    p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    if base is None: base, bfs = f1(y, p), fs
    print(f"{nm:18s} feat {X.shape[1]} | OOF {f1(y, p):.4f} ({f1(y, p) - base:+.4f}) 이긴 fold {int((fs > bfs).sum())}/5 | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]} | 점프 실험 {len(jump_exp)}개 F1 {f1(y[isj], p[isj]):.4f} 오류율 {np.mean((p != y)[isj]):.4f} | 점프 ±12프레임 오류율 {np.mean((p != y)[near]):.3f} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
