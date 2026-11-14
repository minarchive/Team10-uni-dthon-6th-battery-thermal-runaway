"""규칙 안(현재·과거 관측만)의 센서 노이즈 제거 필터 효과를 OOF 5-fold(fast)로 비교. 필터는 실험별·시간순 '뒤쪽 창'만 사용(가운데 정렬/미래 창 금지).
A med3 : 최근 3개 관측의 중앙값    B ema : 지수이동평균(alpha=0.5)    C hampel : 과거 7개 중앙값 기준 |x-med| > 5*1.4826*MAD 이면 중앙값으로 치환
D med3(온도·압력·전압) + 가스는 원본    적용 신호: meas_t1, meas_t2, thermal_t, meas_p, meas_v, atmp_t (+가스 10종)"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
BASE = ["meas_t1", "meas_t2", "thermal_t", "meas_p", "meas_v", "atmp_t"]
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; cls = fr.bat_class.values
eid = raw.experiment_id.values; t_all = raw.relative_time_s.values.astype(float)
groups = pd.Series(np.arange(len(raw))).groupby(eid).indices


def apply(df, cols, kind):
    d = df.copy()
    for e, ix in groups.items():
        ix = ix[np.argsort(t_all[ix], kind="stable")]
        for c in cols:
            s = pd.Series(d[c].values[ix].astype(float))
            if kind == "med3": r = s.rolling(3, min_periods=1).median()
            elif kind == "ema": r = s.ewm(alpha=0.5, adjust=False).mean()
            elif kind == "hampel":
                med = s.rolling(7, min_periods=3).median(); mad = (s - med).abs().rolling(7, min_periods=3).median() * 1.4826 + 1e-6
                r = s.where(((s - med).abs() <= 5 * mad) | med.isna(), med)
            d.iloc[ix, d.columns.get_loc(c)] = r.values
    return d


f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
base = None
for nm, kind, cols in [("원본", None, []), ("A med3 (센서6)", "med3", BASE), ("B ema (센서6)", "ema", BASE), ("C hampel (센서6)", "hampel", BASE), ("D med3 (센서6+가스10)", "med3", BASE + GAS), ("E hampel (센서6+가스10)", "hampel", BASE + GAS)]:
    d = apply(raw, cols, kind) if kind else raw
    F.BASE_SIGNALS = list(F.BASE_SIGNALS); F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    X = F.build_features(d).iloc[lab_i].reset_index(drop=True)
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; m = F.build_model(); m.set_params(n_jobs=16, n_estimators=250, learning_rate=0.06)
        O[va] = m.fit(X[~va], y[~va]).predict_proba(X[va])
    p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    if base is None: base, bfs = f1(y, p), fs
    print(f"{nm:24s} OOF {f1(y, p):.4f} ({f1(y, p) - base:+.4f}) 이긴 fold {int((fs > bfs).sum())}/5 | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
