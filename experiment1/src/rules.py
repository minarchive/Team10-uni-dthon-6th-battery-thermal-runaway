"""프로토콜 후보 규칙의 피처 함수 (모두 인과: 같은 실험의 현재·과거 관측만 사용)."""
import numpy as np
import pandas as pd

T1_L2 = 45.0  # L1->L2 경계: 97/97 실험에서 T1=45.0℃ (데이터에서 관측된 고정 상수, 자유 파라미터 아님)


def _groups(df):
    t = df["relative_time_s"].to_numpy(float)
    for _, ix in df.groupby("experiment_id", sort=False).indices.items():
        yield ix[np.argsort(t[ix], kind="mergesort")]


def f_e1(df: pd.DataFrame) -> pd.DataFrame:
    """E1: T1>=45 도달 이후 경과시간 / 도달 시각 / 도달까지 평균 가열속도 / 도달 여부 / T1-45."""
    n = len(df)
    names = ["e1_cross45", "e1_since45", "e1_tcross45", "e1_rate_to45", "e1_t1_minus45"]
    out = {k: np.full(n, np.nan) for k in names}
    t_all = df["relative_time_s"].to_numpy(float); t1_all = df["meas_t1"].to_numpy(float)
    for ix in _groups(df):
        t, t1 = t_all[ix], t1_all[ix]
        cm = np.fmax.accumulate(t1)
        crossed = cm >= T1_L2
        out["e1_cross45"][ix] = crossed.astype(float)
        out["e1_t1_minus45"][ix] = t1 - T1_L2
        if crossed.any():
            f = int(np.argmax(crossed)); tc = t[f]
            out["e1_since45"][ix[f:]] = t[f:] - tc
            out["e1_tcross45"][ix[f:]] = tc
            out["e1_rate_to45"][ix[f:]] = (T1_L2 - t1[0]) / max(tc - t[0], 5.0)
    return pd.DataFrame(out, index=df.index)


def f_e2(df: pd.DataFrame) -> pd.DataFrame:
    """E2: 45℃ 도달 이후 평균 상승속도(T1,T2,thermal) 와 도달 시점의 T1-T2."""
    n = len(df)
    names = ["e2_rate_t1", "e2_rate_t2", "e2_rate_thermal", "e2_gap_at_cross", "e2_t2_rise_since"]
    out = {k: np.full(n, np.nan) for k in names}
    t_all = df["relative_time_s"].to_numpy(float)
    t1a, t2a, tha = (df[c].to_numpy(float) for c in ("meas_t1", "meas_t2", "thermal_t"))
    for ix in _groups(df):
        t, t1, t2, th = t_all[ix], t1a[ix], t2a[ix], tha[ix]
        crossed = np.fmax.accumulate(t1) >= T1_L2
        if not crossed.any():
            continue
        f = int(np.argmax(crossed))
        since = np.maximum(t[f:] - t[f], 5.0)
        out["e2_rate_t1"][ix[f:]] = (t1[f:] - t1[f]) / since
        out["e2_rate_t2"][ix[f:]] = (t2[f:] - t2[f]) / since
        out["e2_rate_thermal"][ix[f:]] = (th[f:] - th[f]) / since
        out["e2_gap_at_cross"][ix[f:]] = t1[f] - t2[f]
        out["e2_t2_rise_since"][ix[f:]] = t2[f:] - t2[f]
    return pd.DataFrame(out, index=df.index)


def f_e3(df: pd.DataFrame) -> pd.DataFrame:
    """E3: tr_method 별 트리거 신호(1=T1, 2=thermal_t)의 cummax / 상승량 / 30초 기울기 / 10℃ 상승 후 경과시간."""
    n = len(df)
    names = ["e3_trig", "e3_trig_cummax", "e3_trig_rise", "e3_trig_slope30", "e3_trig_since_rise10"]
    out = {k: np.full(n, np.nan) for k in names}
    t_all = df["relative_time_s"].to_numpy(float)
    tr = df["tr_method"].to_numpy(int)
    t1a, tha = df["meas_t1"].to_numpy(float), df["thermal_t"].to_numpy(float)
    for ix in _groups(df):
        s = (t1a if tr[ix[0]] == 1 else tha)[ix]
        t = t_all[ix]
        out["e3_trig"][ix] = s
        out["e3_trig_cummax"][ix] = np.fmax.accumulate(s)
        out["e3_trig_rise"][ix] = s - s[0]
        lag = np.searchsorted(t, t - 30, side="right") - 1
        sl = np.full(len(t), np.nan); ok = lag >= 0
        sl[ok] = (s[ok] - s[lag[ok]]) / np.maximum(t[ok] - t[lag[ok]], 1.0)
        out["e3_trig_slope30"][ix] = sl
        hit = (s - s[0]) >= 10
        if hit.any():
            f = int(np.argmax(hit)); v = np.full(len(t), np.nan); v[f:] = t[f:] - t[f]
            out["e3_trig_since_rise10"][ix] = v
    return pd.DataFrame(out, index=df.index)


V_DROP = 0.7  # L4 시작: 전압이 첫 양수 관측값의 70% 아래로 처음 떨어지는 순간 (EDA: 0.6~0.9 에서 ±2프레임 일치율 81~91%)


def rule_flags(df: pd.DataFrame):
    """행마다 (T1 45℃ 도달 여부, 전압 급락 사건 발생 여부). 모두 인과(현재·과거 관측만)."""
    n = len(df)
    crossed = np.zeros(n, bool); dropped = np.zeros(n, bool)
    t1a, va = df["meas_t1"].to_numpy(float), df["meas_v"].to_numpy(float)
    for ix in _groups(df):
        t1, v = t1a[ix], va[ix]
        crossed[ix] = np.fmax.accumulate(t1) >= T1_L2
        pos = np.nonzero(v > 0)[0]
        if len(pos):
            f = pos[0]; ratio = np.ones(len(v)); ratio[f:] = v[f:] / v[f]
            dropped[ix] = np.fmin.accumulate(ratio) < V_DROP
    return crossed, dropped


def rule_groups(df: pd.DataFrame) -> np.ndarray:
    """상태: 0 = T1<45 (L1), 1 = 도달 후 전압 급락 전 (L2/L3), 2 = 전압 급락 후 (L4~L6)."""
    c, d = rule_flags(df)
    g = np.where(c, np.where(d, 2, 1), 0)
    return g.astype(int)


def apply_rules(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """규칙 제약 디코딩(후처리): L1<->L2 는 T1 규칙, L3<->L4 는 전압 급락 사건으로 고정."""
    pred = np.array(pred).copy(); c, d = rule_flags(df)
    pred[~c] = 1
    pred[c & (pred < 2)] = 2
    pred[c & d & (pred < 4)] = 4
    pred[c & ~d & (pred > 3)] = 3
    return pred


def f_f3(df: pd.DataFrame) -> pd.DataFrame:
    """F3: 사건 시계 피처 (전압 급락, V/V0<0.3, T1>=500, T1 1프레임 +20, T2>=400 이후 경과시간 등)."""
    n = len(df)
    names = ["f3_vcmin", "f3_drop_since", "f3_v03_since", "f3_t1_500_since", "f3_dt1_20_since", "f3_t2_400_since"]
    out = {k: np.full(n, np.nan) for k in names}
    t_all = df["relative_time_s"].to_numpy(float)
    t1a, t2a, va = df["meas_t1"].to_numpy(float), df["meas_t2"].to_numpy(float), df["meas_v"].to_numpy(float)

    def since(t, flag):
        o = np.full(len(t), np.nan)
        if flag.any():
            f = int(np.argmax(flag)); o[f:] = t[f:] - t[f]
        return o
    for ix in _groups(df):
        t, t1, t2, v = t_all[ix], t1a[ix], t2a[ix], va[ix]
        pos = np.nonzero(v > 0)[0]
        ratio = np.ones(len(v))
        if len(pos):
            f = pos[0]; ratio[f:] = v[f:] / v[f]
        cmin = np.fmin.accumulate(ratio)
        out["f3_vcmin"][ix] = cmin
        out["f3_drop_since"][ix] = since(t, cmin < V_DROP)
        out["f3_v03_since"][ix] = since(t, cmin < 0.3)
        out["f3_t1_500_since"][ix] = since(t, np.fmax.accumulate(t1) >= 500)
        d1 = np.diff(t1, prepend=t1[0])
        out["f3_dt1_20_since"][ix] = since(t, np.fmax.accumulate(d1) >= 20)
        out["f3_t2_400_since"][ix] = since(t, np.fmax.accumulate(t2) >= 400)
    return pd.DataFrame(out, index=df.index)


FEATURE_FNS = {"E1": f_e1, "E2": f_e2, "E3": f_e3, "F3": f_f3}


def check_causal(fn, df: pd.DataFrame, n_exp: int = 4, frac: float = 0.6, **kw) -> bool:
    """인과성 검사: 각 실험을 앞 frac 만 남기고 계산한 값이, 전체로 계산한 값의 같은 행과 동일해야 한다."""
    ids = df["experiment_id"].drop_duplicates().head(n_exp)
    full_df = df[df["experiment_id"].isin(ids)].reset_index(drop=True)
    trunc_parts = []
    for _, g in full_df.groupby("experiment_id", sort=False):
        g = g.sort_values("relative_time_s", kind="mergesort")
        trunc_parts.append(g.iloc[: max(5, int(len(g) * frac))])
    trunc = pd.concat(trunc_parts)
    a = fn(full_df, **kw)
    a.index = full_df["sample_id"].values
    b = fn(trunc.reset_index(drop=True), **kw)
    b.index = trunc["sample_id"].values
    a = a.loc[b.index]
    return bool(np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=1e-6, atol=1e-6))
