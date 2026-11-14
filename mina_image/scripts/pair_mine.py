"""신호 쌍 채굴: 방식별(가열/과충전)로 L2/L3 프레임에서 파생 신호(쌍의 차이, 첫 관측 대비, 기울기 차이)의 누적 극값이 L3 를 얼마나 가르는지 AUC 로 선별(풀링 AUC: 실험을 섞음 / 실험 내 AUC 평균: 방향 일관성).
모두 인과적(과거만): 누적 극값, 첫 관측 대비, 30초 기울기. atmp 는 글리치 때문에 실험 첫 관측값(atmp0)으로 대체."""
import sys, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata
from common import DATA
tr_sel = int(sys.argv[1]); GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation"]], ignore_index=True)
raw["experiment_id"] = raw.experiment_id.astype(str); raw = raw[raw.tr_method == tr_sel].sort_values(["experiment_id", "relative_time_s"], kind="stable").reset_index(drop=True)
eid = raw.experiment_id.values; t = raw.relative_time_s.values.astype(float); idx = pd.Series(np.arange(len(raw))).groupby(eid).indices
def first(x): o = np.empty(len(x)); [o.__setitem__(ix, x[ix][0]) for ix in idx.values()]; return o
def slope(x, w=30):
    o = np.full(len(x), np.nan)
    for ix in idx.values():
        tt = t[ix]; j = np.searchsorted(tt, tt - w, side="right") - 1; ok = j >= 0; v = np.full(len(ix), np.nan); v[ok] = (x[ix][ok] - x[ix][j[ok]]) / np.maximum(tt[ok] - tt[j[ok]], 1); o[ix] = v
    return o
B = {"t1": raw.meas_t1.values.astype(float), "t2": raw.meas_t2.values.astype(float), "th": raw.thermal_t.values.astype(float), "amb0": first(raw.atmp_t.values.astype(float)),
     "p": raw.meas_p.values.astype(float), "v": raw.meas_v.values.astype(float), "gas": np.log1p(raw[GAS].sum(axis=1).values)}
temps = ["t1", "t2", "th", "amb0"]
S = {}
for a_, b_ in itertools.combinations(temps, 2): S[f"{a_}-{b_}"] = B[a_] - B[b_]
for k in ("t1", "t2", "th", "p", "v", "gas"): S[f"{k}_rel"] = B[k] - first(B[k]); S[f"{k}_slope30"] = slope(B[k])
for a_, b_ in itertools.combinations(["t1", "t2", "th"], 2): S[f"slope({a_})-slope({b_})"] = slope(B[a_]) - slope(B[b_]); S[f"rel({a_})-rel({b_})"] = (B[a_] - first(B[a_])) - (B[b_] - first(B[b_]))
S["v/v0"] = B["v"] / np.where(first(B["v"]) > 0, first(B["v"]), np.nan); S["th-amb0_rel"] = (B["th"] - B["amb0"]) - first(B["th"] - B["amb0"]); S["t1-amb0"] = B["t1"] - B["amb0"]
for ix_name, ix in [("th-amb0", B["th"] - B["amb0"])]: S[ix_name] = ix
m23 = raw.label.isin([2, 3]).values; y = (raw.label.values == 3).astype(int)
def auc(s, yy):
    ok = ~np.isnan(s); s, yy = s[ok], yy[ok]
    if yy.sum() == 0 or yy.sum() == len(yy): return np.nan
    r = rankdata(s); return (r[yy == 1].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (len(yy) - yy.sum()))
rows = []
for nm, s in S.items():
    ser = pd.Series(s)
    for kind, cm in [("cummax", ser.groupby(eid).cummax().values), ("cummin", ser.groupby(eid).cummin().values)]:
        pa = auc(cm[m23], y[m23]); wl = []
        for e, ix in idx.items():
            mm = ix[m23[ix]]
            if len(mm) > 10 and 0 < y[mm].sum() < len(mm): wl.append(auc(cm[mm], y[mm]))
        rows.append((nm, kind, pa, np.nanmean(wl) if wl else np.nan, len(wl)))
r = pd.DataFrame(rows, columns=["파생 신호", "누적", "풀링 AUC", "실험내 AUC 평균", "실험수"]); r["|AUC-0.5|"] = (r["풀링 AUC"] - 0.5).abs()
print(f"\n=== {'과충전' if tr_sel == 2 else '가열'} (실험 {len(idx)}개, L2/L3 프레임 {int(m23.sum())}, L3 비율 {y[m23].mean():.2f}) | 후보 {len(r)}개")
print(r.sort_values("|AUC-0.5|", ascending=False).head(14).drop(columns="|AUC-0.5|").round(3).to_string(index=False))
