"""센서값으로 정의한 이벤트(전압 하락, 전압 0, 압력 상승, 온도 급상승, 최고온도 도달 등)를 여러 임계값으로 만들어 단계 k(2..6) 시작 시각과의 일치도를 본다.
일치도 = |이벤트 시각 - 단계 시작 시각| <= 10초인 실험 비율. 탐색 분석용(사후 정렬 허용: 단계가 어떤 센서 사건과 맞닿는지 파악이 목적).
상위 쌍은 이후 '인과적' 피처(이벤트 이후 경과 시간)로 바꿔 모델에 넣는다."""
import numpy as np, pandas as pd
from common import DATA
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation"]], ignore_index=True)
raw["experiment_id"] = raw.experiment_id.astype(str)
GAS = [c for c in raw.columns if c.startswith("meas_") and c not in ("meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p", "meas_t1_plus_t2")]
res = []
for e, g in raw.sort_values(["experiment_id", "relative_time_s"]).groupby("experiment_id"):
    g = g.reset_index(drop=True); t = g.relative_time_s.values.astype(float); l = g.label.values
    on = {k: t[l >= k].min() for k in range(2, 7) if (l >= k).any()}
    v = g.meas_v.values.astype(float); v0 = max(v[0], 1e-6); vr = v / v0
    p = g.meas_p.values.astype(float); t1 = g.meas_t1.values.astype(float); th = g.thermal_t.values.astype(float); gas = np.log1p(g[GAS].sum(axis=1).values)
    dt = np.r_[np.nan, np.diff(t)]
    first = lambda mask: (t[np.argmax(mask)] if mask.any() else np.nan)
    ev = {}
    ev["v==0"] = first(v <= 0.01)
    for k in (0.02, 0.05, 0.1, 0.2, 0.5): ev[f"v/v0<={1 - k:.2f}"] = first(vr <= 1 - k)
    dv = np.r_[0, np.diff(vr)]
    for k in (0.03, 0.1, 0.3): ev[f"v 한 칸 하락>={k}"] = first(-dv >= k)
    for th_ in (0.3, 1, 2, 5, 10, 20): ev[f"p-p0>={th_}"] = first(p - p[0] >= th_)
    dp = np.r_[0, np.diff(p)]
    for th_ in (0.2, 0.5, 1, 3): ev[f"p 한 칸 상승>={th_}"] = first(dp >= th_)
    for th_ in (60, 80, 100, 150, 200, 300): ev[f"T1>={th_}"] = first(t1 >= th_)
    s30 = np.full(len(t), np.nan)
    j = np.searchsorted(t, t - 30, side="right") - 1; ok = j >= 0; s30[ok] = (t1[ok] - t1[j[ok]]) / np.maximum(t[ok] - t[j[ok]], 1)
    for th_ in (0.1, 0.3, 0.6, 1.0): ev[f"T1 기울기(30s)>={th_}"] = first(s30 >= th_)
    d1 = np.r_[0, np.diff(t1)]
    for th_ in (2, 5, 10): ev[f"T1 한 칸 상승>={th_}"] = first(d1 >= th_)
    cm = np.maximum.accumulate(t1)
    for th_ in (2, 5, 20): ev[f"T1 최고점에서 {th_} 하락"] = first((cm - t1 >= th_) & (cm >= 80))
    ev["T1 전체 최댓값 시각(사후)"] = t[int(np.argmax(t1))]
    for th_ in (2, 5): ev[f"thermal_t 한 칸 상승>={th_}"] = first(np.r_[0, np.diff(th)] >= th_)
    ev["thermal_t>=60"] = first(th >= 60); ev["thermal_t>=100"] = first(th >= 100)
    for q in (1.0, 3.0): ev[f"gas>={q}"] = first(gas >= q)
    for en, et in ev.items():
        for k, ot in on.items(): res.append((en, k, et - ot if not np.isnan(et) else np.nan))
d = pd.DataFrame(res, columns=["event", "stage", "lag"])
rows = []
for (en, k), g in d.groupby(["event", "stage"]):
    x = g.lag.values; cov = np.mean(~np.isnan(x)); xx = x[~np.isnan(x)]
    rows.append((en, k, len(x), cov, np.mean(np.abs(xx) <= 10) * cov if len(xx) else 0, np.median(xx) if len(xx) else np.nan, np.percentile(np.abs(xx), 75) if len(xx) else np.nan))
r = pd.DataFrame(rows, columns=["event", "stage", "n", "coverage", "within10s", "median_lag_s", "abs_lag_p75"])
for k in range(2, 7):
    print(f"\n=== 단계 L{k} 시작과 가장 잘 겹치는 이벤트 (전체 실험 대비 ±10초 이내 비율)")
    print(r[r.stage == k].sort_values("within10s", ascending=False).head(6)[["event", "within10s", "coverage", "median_lag_s", "abs_lag_p75"]].round(3).to_string(index=False))
