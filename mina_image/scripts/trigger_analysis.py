"""L3 시작 트리거 탐색: tr_method 별로, 신호마다 '누적 극값이 임계값을 처음 넘은 뒤 = L3' 인과 규칙을 만들고 L2/L3 프레임에서 이진 F1 로 비교.
신호는 실험 첫 관측 대비(상대값)로도 본다. 임계값은 해당 그룹 L3 시작 시점 값의 분포(5~95%)에서 격자로 고른다(그룹 내 최적이라 낙관적: 어떤 신호가 가능성이 있는지 보는 용도).
기준선: 현재 센서 OOF 의 같은 프레임 이진 F1."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.sort_values(["experiment_id", "relative_time_s"]).reset_index(drop=True)
g = raw.groupby("experiment_id")
raw["gas_sum"] = raw[GAS].sum(axis=1)
S = {}
for c in ["meas_t1", "meas_t2", "thermal_t", "meas_p", "gas_sum", "atmp_t"] + GAS: S[c] = raw[c]
S["meas_v"] = raw.meas_v / g.meas_v.transform("first").replace(0, np.nan)
S["t1_minus_amb"] = raw.meas_t1 - raw.atmp_t
S["thermal_minus_t1"] = raw.thermal_t - raw.meas_t1
for c in ["meas_t1", "meas_t2", "thermal_t", "meas_p", "gas_sum"]: S[c + "_rel"] = raw[c] - g[c].transform("first")
S["t1_slope60"] = (raw.meas_t1 - g.meas_t1.shift(12)) / 60
tb = table(); so = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy")
lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
sp = pd.Series(F.postprocess(tb[lab].assign(experiment_id=tb.experiment_id[lab].astype(str)).reset_index(drop=True), so, 0.7), index=tb.sample_id[lab].astype(str).values)
raw["sens_pred"] = sp.loc[raw.sample_id].values
m23 = raw.label.isin([2, 3]).values
for tr, nm in [(1, "가열(tr_method=1)"), (2, "과충전(tr_method=2)")]:
    gm = (raw.tr_method == tr).values
    fm = m23 & gm
    y = (raw.label[fm] == 3).astype(int).values
    onset = raw[gm & (raw.label == 3)].groupby("experiment_id").head(1)
    res = []
    for name, s in S.items():
        s_ = s.copy()
        if name == "meas_v": cm = s_.groupby(raw.experiment_id).cummin(); sgn = -1   # 전압은 하락이 신호
        else: cm = s_.groupby(raw.experiment_id).cummax(); sgn = 1
        ov = (s_[onset.index]).dropna()
        if len(ov) < 5: continue
        grid = np.unique(np.nanpercentile(ov, np.linspace(2, 98, 25)))
        for th in grid:
            pred = ((cm[fm] <= th) if sgn < 0 else (cm[fm] >= th)).astype(int).values
            res.append((name, th, f1_score(y, pred, zero_division=0)))
    d = pd.DataFrame(res, columns=["signal", "thr", "L3 F1"]).sort_values("L3 F1", ascending=False).drop_duplicates("signal")
    base = f1_score(y, (raw.sens_pred[fm] >= 3).astype(int).values, zero_division=0)
    n_exp = raw[gm].experiment_id.nunique()
    print(f"\n=== {nm}: 실험 {n_exp}개, L2/L3 프레임 {fm.sum()}, L3 비율 {y.mean():.2f} | 현재 센서(OOF) L3 F1 {base:.4f}")
    print("  신호별 최적 단일 임계 규칙(누적 극값 기준) 상위 12:")
    for _, r in d.head(12).iterrows(): print(f"    {r.signal:20s} 임계 {r.thr:10.3f}  L3 F1 {r['L3 F1']:.4f}")
    # 시작 시점 값의 분산(변동계수): 값이 일정할수록 트리거 후보
    cv = {k: (v.std() / (abs(v.mean()) + 1e-9)) for k, v in ((k, s[onset.index].dropna()) for k, s in S.items()) if len(v) > 5}
    print("  L3 시작 시점 값의 변동계수(작을수록 일정한 트리거):", {k: round(v, 2) for k, v in sorted(cv.items(), key=lambda x: x[1])[:6]})
