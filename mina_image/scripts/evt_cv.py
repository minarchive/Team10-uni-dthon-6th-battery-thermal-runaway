"""이벤트 기반 인과 피처(이벤트가 처음 발생한 뒤 경과 시간, 발생 전 NaN)를 X1 피처에 추가해 OOF 5-fold(fast)로 비교. 모든 이벤트는 같은 실험의 현재·과거 관측만으로 판정.
전압: 시작값 대비 비율이 처음 임계 이하 / 전압 0 / 한 칸 하락.  온도: T1 한 칸 상승, 30초 기울기, 최고점(>=80°C 이후) 대비 하락.  압력: 한 칸 상승, 시작값 대비 상승.  열화상 온도: 한 칸 상승."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
eid = raw.experiment_id.values; t_all = raw.relative_time_s.values.astype(float)
cols = {}
def put(name, ix, arr): cols.setdefault(name, np.full(len(raw), np.nan))[ix] = arr
for e, ix in pd.Series(np.arange(len(raw))).groupby(eid).indices.items():
    ix = ix[np.argsort(t_all[ix], kind="stable")]; t = t_all[ix]; n = len(ix)
    v = raw.meas_v.values[ix].astype(float); v0 = max(v[0], 1e-6); vr = v / v0
    p = raw.meas_p.values[ix].astype(float); t1 = raw.meas_t1.values[ix].astype(float); th = raw.thermal_t.values[ix].astype(float)
    def since(mask, name):
        j = np.maximum.accumulate(mask.astype(int))      # 처음 발생 이후 True
        tf = t[np.argmax(mask)] if mask.any() else np.inf
        put(name, ix, np.where(t >= tf, t - tf, np.nan))
    cmv = np.minimum.accumulate(vr)
    for k in (0.98, 0.95, 0.9, 0.8, 0.5, 0.2): since(cmv <= k, f"ev_v_le{int(k * 100)}")
    since(v <= 0.01, "ev_v_zero")
    dv = np.r_[0, np.diff(vr)]
    for k in (0.03, 0.1, 0.3): since(-dv >= k, f"ev_vstep{int(k * 100)}")
    d1 = np.r_[0, np.diff(t1)]
    for k in (5, 10): since(d1 >= k, f"ev_t1step{k}")
    j = np.searchsorted(t, t - 30, side="right") - 1; ok = j >= 0; s30 = np.full(n, np.nan); s30[ok] = (t1[ok] - t1[j[ok]]) / np.maximum(t[ok] - t[j[ok]], 1)
    for k in (0.6, 1.0): since(np.nan_to_num(s30, nan=0) >= k, f"ev_t1slope{int(k * 10)}")
    cm = np.maximum.accumulate(t1)
    for k in (2, 5, 20): since((cm - t1 >= k) & (cm >= 80), f"ev_t1drop{k}")
    dp = np.r_[0, np.diff(p)]
    for k in (0.5, 1, 3): since(dp >= k, f"ev_pstep{int(k * 10)}")
    for k in (2, 5, 10): since(p - p[0] >= k, f"ev_prise{k}")
    dth = np.r_[0, np.diff(th)]
    for k in (2, 5): since(dth >= k, f"ev_thstep{k}")
EV = pd.DataFrame(cols)
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; cls = fr.bat_class.values
X1 = pd.read_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet"); EVl = EV.iloc[lab_i].reset_index(drop=True)
print(f"이벤트 피처 {EV.shape[1]}개", flush=True)
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=16, verbose=-1, random_state=42)
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
base = None
for nm, X in [("X1 기준", X1), ("X1 + 이벤트 피처", pd.concat([X1, EVl], axis=1))]:
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; O[va] = lgb.LGBMClassifier(**P0).fit(X[~va], y[~va]).predict_proba(X[va])
    p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    if base is None: base, bfs = f1(y, p), fs
    print(f"{nm:16s} feat {X.shape[1]} | OOF {f1(y, p):.4f} ({f1(y, p) - base:+.4f}) 이긴 fold {int((fs > bfs).sum())}/5 | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
