"""사건(event) 기반 인과 피처 + 인과 클리닝.

라벨 경계가 절대 온도가 아니라 '사건의 진행'(전압 붕괴, 온도 급상승, 압력 급변, 가스 방출)에 묶여 있어,
프레임 한 장의 값 대신 '사건이 일어난 뒤 경과 시간', '마지막 신고점 갱신 후 경과 시간' 같은 상태 변수를 만든다.
모든 계산은 같은 experiment_id 의 현재·과거 관측만 사용한다 (cummax/cummin/ffill 은 과거 방향).
"""
import numpy as np
import pandas as pd

GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]


def clean_causal(df: pd.DataFrame) -> pd.DataFrame:
    """고정 물리 범위를 벗어난 값을 NaN 처리 후 '과거' 마지막 유효값으로 채움 (미래 값 미사용)."""
    d = df.sort_values(["experiment_id", "relative_time_s"], kind="mergesort").copy()
    d.loc[(d.atmp_t > 60) | (d.atmp_t < 5), "atmp_t"] = np.nan          # 실험실 온도 글리치(예: 777)
    d.loc[d.meas_t1 <= 1, "meas_t1"] = np.nan                             # 센서 단선/드롭아웃
    d.loc[d.meas_t2 <= 1, "meas_t2"] = np.nan
    cols = ["atmp_t", "meas_t1", "meas_t2"]
    d[cols] = d.groupby("experiment_id")[cols].ffill()                    # 과거 방향 채움만
    return d.reindex(df.index)


def _since(t, flag):
    """flag(bool 배열, 시간순)가 처음 True 가 된 시점 이후 경과 시간. 아직이면 NaN. 한 번 켜지면 유지(과거만 사용)."""
    first = np.argmax(flag) if flag.any() else None
    out = np.full(len(t), np.nan)
    if first is not None:
        out[first:] = t[first:] - t[first]
    return out


def event_features(df: pd.DataFrame) -> pd.DataFrame:
    base = df[["experiment_id", "relative_time_s"] + ["meas_v", "meas_t1", "meas_t2", "meas_p", "thermal_t"] + GAS].copy()
    out = pd.DataFrame(index=df.index)
    names = ["ev_v_ratio", "ev_v_min_ratio", "ev_v_collapse_since", "ev_t1_rise", "ev_t1_since_rise10", "ev_t1_since_rise25", "ev_t1_since_rise50",
             "ev_t1_since_abs80", "ev_t1_since_abs120", "ev_t1_since_abs200", "ev_t1_since_newhigh", "ev_t1_slope_max", "ev_t1_since_slopemax",
             "ev_p_rise", "ev_p_since_rise3", "ev_p_since_rise10", "ev_p_since_pmax", "ev_gas_cum", "ev_co2_since10", "ev_hf_since1",
             "ev_thermal_since_newhigh", "ev_t2_since_rise25"]
    buf = {n: np.full(len(df), np.nan) for n in names}
    for _, ix in base.groupby("experiment_id", sort=False).indices.items():
        sub = base.iloc[ix]
        o = np.argsort(sub.relative_time_s.to_numpy(float), kind="mergesort")
        ix = ix[o]; s = base.iloc[ix]
        t = s.relative_time_s.to_numpy(float)
        v = s.meas_v.to_numpy(float); t1 = pd.Series(s.meas_t1.to_numpy(float)).ffill().to_numpy(); t2 = pd.Series(s.meas_t2.to_numpy(float)).ffill().to_numpy()
        p = s.meas_p.to_numpy(float); th = s.thermal_t.to_numpy(float)
        v0 = v[np.nonzero(v > 0)[0][0]] if (v > 0).any() else 1.0
        v0 = v0 if v0 > 0 else 1.0
        # 첫 유효 전압을 '과거'로만 쓰려면 첫 양수 관측 이전엔 비율 NaN
        vr = v / v0
        first_pos = np.nonzero(v > 0)[0][0] if (v > 0).any() else len(v)
        vr[:first_pos] = np.nan
        buf["ev_v_ratio"][ix] = vr
        vmin = np.fmin.accumulate(np.where(np.isnan(vr), 9, vr)); vmin[np.isnan(vr)] = np.nan
        buf["ev_v_min_ratio"][ix] = vmin
        buf["ev_v_collapse_since"][ix] = _since(t, np.nan_to_num(vr, nan=1.0) < 0.5)
        r1 = t1 - t1[0]; buf["ev_t1_rise"][ix] = r1
        for k in (10, 25, 50): buf[f"ev_t1_since_rise{k}"][ix] = _since(t, r1 >= k)
        for k in (80, 120, 200): buf[f"ev_t1_since_abs{k}"][ix] = _since(t, t1 >= k)
        cm = np.fmax.accumulate(t1); newhigh = np.r_[True, cm[1:] > cm[:-1] + 0.05]
        last = np.maximum.accumulate(np.where(newhigh, t, -1.0)); buf["ev_t1_since_newhigh"][ix] = t - last
        # 30초 기울기(과거 구간만)
        lag = np.searchsorted(t, t - 30, side="right") - 1
        sl = np.where(lag >= 0, (t1 - t1[np.clip(lag, 0, None)]) / np.maximum(t - t[np.clip(lag, 0, None)], 1), np.nan)
        slm = np.fmax.accumulate(np.nan_to_num(sl, nan=-1e9)); slm[slm < -1e8] = np.nan
        buf["ev_t1_slope_max"][ix] = slm
        am = np.array([np.nanargmax(np.nan_to_num(sl[:i + 1], nan=-1e9)) for i in range(len(t))]) if len(t) < 3000 else np.zeros(len(t), int)
        buf["ev_t1_since_slopemax"][ix] = t - t[am]
        rp = p - p[0]; buf["ev_p_rise"][ix] = rp
        for k in (3, 10): buf[f"ev_p_since_rise{k}"][ix] = _since(t, rp >= k)
        pm = np.fmax.accumulate(p); pnew = np.r_[True, pm[1:] > pm[:-1] + 1e-6]
        buf["ev_p_since_pmax"][ix] = t - np.maximum.accumulate(np.where(pnew, t, -1.0))
        gas = s[GAS].to_numpy(float).sum(1); dt = np.r_[5.0, np.diff(t)]
        buf["ev_gas_cum"][ix] = np.log1p(np.cumsum(gas * dt))
        buf["ev_co2_since10"][ix] = _since(t, s.meas_co2.to_numpy(float) >= 10)
        buf["ev_hf_since1"][ix] = _since(t, s.meas_hf.to_numpy(float) >= 1)
        thm = np.fmax.accumulate(th); thn = np.r_[True, thm[1:] > thm[:-1] + 0.05]
        buf["ev_thermal_since_newhigh"][ix] = t - np.maximum.accumulate(np.where(thn, t, -1.0))
        buf["ev_t2_since_rise25"][ix] = _since(t, (t2 - t2[0]) >= 25)
    return pd.DataFrame(buf, index=df.index)
