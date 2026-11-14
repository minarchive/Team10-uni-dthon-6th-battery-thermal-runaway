"""과충전(tr_method=2)에서 L3 시작을 가르는 신호 탐색. 파생 신호 = 단일/차이/비율/하락량/기울기/이벤트 이후 경과. 규칙 = 누적 극값(cummax 또는 cummin)이 임계를 넘으면 L3.
평가: L2/L3 프레임의 이진 F1, 임계값은 실험 하나를 빼고(LOEO) 나머지 실험에서 최적화한 값으로 그 실험에 적용(과적합 방지). 비교: 센서 LightGBM OOF 의 같은 프레임 이진 F1.
--tr 로 방식 선택(2=과충전, 1=가열)."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
tr_sel = int(sys.argv[1]) if len(sys.argv) > 1 else 2
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw[raw.tr_method == tr_sel].sort_values(["experiment_id", "relative_time_s"], kind="stable").reset_index(drop=True)
tb = table(); so = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy"); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
sp = pd.Series(so[:, 2:].sum(1), index=tb.sample_id.astype(str).values[lab]); raw["sens_p3"] = sp.loc[raw.sample_id].values
S = {}; g = raw.groupby("experiment_id"); t = raw.relative_time_s.values.astype(float)
def sl(c, w=30):
    out = np.full(len(raw), np.nan)
    for _, ix in g.indices.items():
        tt = t[ix]; j = np.searchsorted(tt, tt - w, side="right") - 1; ok = j >= 0; x = raw[c].values[ix].astype(float); o = np.full(len(ix), np.nan); o[ok] = (x[ok] - x[j[ok]]) / np.maximum(tt[ok] - tt[j[ok]], 1); out[ix] = o
    return out
v0 = g.meas_v.transform("first").replace(0, np.nan)
for c in ["meas_t1", "meas_t2", "thermal_t", "atmp_t", "meas_p", "meas_v"]: S[c] = raw[c].values.astype(float)
S["t2-t1"] = (raw.meas_t2 - raw.meas_t1).values; S["thermal-t1"] = (raw.thermal_t - raw.meas_t1).values; S["thermal-t2"] = (raw.thermal_t - raw.meas_t2).values; S["t1-amb"] = (raw.meas_t1 - raw.atmp_t).values
S["thermal-amb"] = (raw.thermal_t - raw.atmp_t).values; S["v/v0"] = (raw.meas_v / v0).values; S["v-vmax"] = (raw.meas_v - g.meas_v.cummax()).values; S["p-p0"] = (raw.meas_p - g.meas_p.transform("first")).values
S["t1_rel"] = (raw.meas_t1 - g.meas_t1.transform("first")).values; S["thermal_rel"] = (raw.thermal_t - g.thermal_t.transform("first")).values; S["t2_rel"] = (raw.meas_t2 - g.meas_t2.transform("first")).values
S["gas_sum"] = np.log1p(raw[GAS].sum(axis=1)).values
for c in GAS: S[c] = raw[c].values.astype(float)
for c, nm in [("meas_t1", "t1"), ("thermal_t", "thermal"), ("meas_p", "p"), ("meas_v", "v"), ("meas_t2", "t2")]: S[f"{nm}_slope30"] = sl(c)
S["t1_cummax_gap"] = (g.meas_t1.cummax() - raw.meas_t1).values
cm45 = g.meas_t1.cummax().values >= 45; t45 = pd.Series(np.where(cm45, t, np.nan)).groupby(raw.experiment_id.values).transform("min").values; S["time_since_T1_45"] = np.where(cm45, t - t45, np.nan)   # 도달 이후 행에서만(미래 도달 시각 미사용)
tv = pd.Series(np.where(raw.meas_v.values >= g.meas_v.cummax().values - 1e-9, t, np.nan)); S["time_since_vmax"] = t - tv.groupby(raw.experiment_id.values).ffill().values
m23 = raw.label.isin([2, 3]).values; y = (raw.label.values == 3).astype(int); e = raw.experiment_id.values; ue = np.unique(e[m23])
sens_bin = f1_score(y[m23], (raw.sens_p3.values[m23] >= 0.5).astype(int), zero_division=0)
res = []
for name, s in S.items():
    s = pd.Series(s); cmx = s.groupby(raw.experiment_id.values).cummax().values; cmn = s.groupby(raw.experiment_id.values).cummin().values
    for kind, cm, sign in [("cummax>=", cmx, 1), ("cummin<=", cmn, -1)]:
        pred_all = np.zeros(len(raw), int); ok = True
        for ho in ue:
            trm = m23 & (e != ho); tem = m23 & (e == ho); v_ = cm[trm]; v_ = v_[~np.isnan(v_)]
            if len(v_) < 50: ok = False; break
            grid = np.unique(np.nanpercentile(v_, np.linspace(2, 98, 33))); best, bt = -1, grid[0]
            for th in grid:
                pr = ((cm[trm] >= th) if sign > 0 else (cm[trm] <= th)) & ~np.isnan(cm[trm]); f = f1_score(y[trm], pr.astype(int), zero_division=0)
                if f > best: best, bt = f, th
            pred_all[tem] = (((cm[tem] >= bt) if sign > 0 else (cm[tem] <= bt)) & ~np.isnan(cm[tem])).astype(int)
        if ok: res.append((name, kind, f1_score(y[m23], pred_all[m23], zero_division=0)))
r = pd.DataFrame(res, columns=["신호", "규칙", "LOEO L3 F1"]).sort_values("LOEO L3 F1", ascending=False).drop_duplicates("신호")
print(f"{'과충전' if tr_sel == 2 else '가열'}: 실험 {len(ue)}개, L2/L3 프레임 {int(m23.sum())}, L3 비율 {y[m23].mean():.2f} | 센서 LightGBM(OOF) 같은 프레임 L3 F1 {sens_bin:.4f}")
print("단일 규칙(임계 1개, LOEO) 상위 15:"); print(r.head(15).round(4).to_string(index=False))
