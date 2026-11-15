"""새 피처 세트 (처음부터 설계). 같은 실험의 현재·과거 행만 사용. 시간 창은 초 단위.

제외: 가스 10종(노이즈), atmp_t(단계와 무관), meas_a(상수), meas_t1_plus_t2(중복), indu_t.1(결측이 많아 사용하지 않음)
"""
import numpy as np, pandas as pd

META = ['bat_class', 'tr_method', 'soc_status', 'bat_cc']


def _lag(d, col, w):
    """같은 실험에서 t-w초 이전(이하) 마지막 관측값"""
    left = pd.DataFrame({'k': d.t - w, 'e': d.e, 'i': np.arange(len(d))}).sort_values('k')
    right = d[['t', 'e', col]].rename(columns={'t': 'k'}).sort_values('k')
    return pd.merge_asof(left, right, on='k', by='e', direction='backward').sort_values('i')[col].values


GAS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']


def _p23(frame):
    """2→3 전용: 압력 급등 시작 + 가열 가속 (같은 실험·현재 이하)"""
    d = pd.DataFrame({'sid': frame.sample_id.astype(str).values, 'e': frame.experiment_id.astype(str).values, 't': frame.relative_time_s.astype(float).values,
                      'p': pd.to_numeric(frame.meas_p, errors='coerce').values, 't1': pd.to_numeric(frame.meas_t1, errors='coerce').values})
    d = d.sort_values(['e', 't']).reset_index(drop=True); F = {}
    for w in (5, 10):
        r = pd.Series((d.p.values - _lag(d, 'p', w)) / w); F[f'p_r{w}'] = r
        g = r.groupby(d.e); mu = g.transform(lambda v: v.shift(1).expanding(5).mean()); sd = g.transform(lambda v: v.shift(1).expanding(5).std())
        z = (r - mu) / (sd + 1e-6); F[f'p_r{w}_z'] = z; zc = z.groupby(d.e).cummax()
        for th in (2, 3, 4): F[f'p_r{w}_since_z{th}'] = d.t - d.t.where(zc >= th).groupby(d.e).cummin()
    t1c = d.t1.groupby(d.e).cummax()
    for th in (45, 50):
        hit = t1c >= th; p_at = d.p.where(hit & ~hit.groupby(d.e).shift(1, fill_value=False)).groupby(d.e).ffill()
        F[f'p_rise_since_t1_{th}'] = d.p - p_at
    r10 = pd.Series((d.t1.values - _lag(d, 't1', 10)) / 10); hit45 = t1c >= 45
    r_at45 = r10.where(hit45 & ~hit45.groupby(d.e).shift(1, fill_value=False)).groupby(d.e).ffill()
    F['t1_rate_rel45'] = r10 / r_at45.where(r_at45.abs() > 1e-3)
    X = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan).astype('float32'); X.index = d.sid.values
    return X.loc[frame.sample_id.astype(str).values]


def build(frame, mode='compact'):
    if mode in ('rich_z', 'rich_z_p23'):                               # rich에 실험 기준 z-score를 '추가' (아무것도 빼지 않음)
        X = build(frame, 'rich'); Z = _rich(frame, rel=True); Z = Z[[c for c in Z.columns if '_z10' in c]]
        X = pd.concat([X, Z], axis=1)
        return pd.concat([X, _p23(frame)], axis=1) if mode == 'rich_z_p23' else X
    if mode == 'rich_rel_p23':
        return pd.concat([build(frame, 'rich_rel'), _p23(frame)], axis=1)
    """mode: compact | rich (윈도우 통계 추가) | rich_gas (+평활 가스 합) | rich_rel (실험 기준 표현: p·thermal·indu 절대값 제거, z-score 추가)"""
    X = _build_compact(frame)
    if mode == 'compact':
        return X
    R = _rich(frame, gas=(mode == 'rich_gas'), rel=(mode == 'rich_rel'))
    X = pd.concat([X, R], axis=1)
    if mode == 'rich_rel':
        X = X[[c for c in X.columns if not any(c == f'{s}' or c.startswith(f'{s}_m') or c == f'{s}_cmax' for s in ('meas_p', 'thermal_t', 'indu_t'))]]
    return X


def _rich(frame, gas=False, rel=False):
    d = pd.DataFrame({'sid': frame.sample_id.astype(str).values, 'e': frame.experiment_id.astype(str).values, 't': frame.relative_time_s.astype(float).values})
    sig = ['meas_t1', 'meas_t2', 'thermal_t', 'meas_p', 'meas_v', 'indu_t']
    for c in sig: d[c] = pd.to_numeric(frame[c], errors='coerce').values
    if gas: d['gas'] = frame[GAS].apply(pd.to_numeric, errors='coerce').sum(1).values; sig = sig + ['gas']
    d = d.sort_values(['e', 't']).reset_index(drop=True); d['_ts'] = pd.to_datetime(d.t, unit='s'); F = {}
    gb = d.groupby('e', sort=False)
    for w in (30, 120, 300):
        r = gb.rolling(f'{w}s', on='_ts')
        for s in sig:
            F[f'{s}_w{w}_mean'] = r[s].mean().reset_index(drop=True); F[f'{s}_w{w}_std'] = r[s].std().reset_index(drop=True)
            F[f'{s}_w{w}_min'] = r[s].min().reset_index(drop=True); F[f'{s}_w{w}_max'] = r[s].max().reset_index(drop=True)
    for s in sig:
        x = d[s]; cm = gb[s].cummax(); peak_t = d.t.where(x.eq(cm) & x.notna()).groupby(d.e).ffill()
        F[f'{s}_since_peak'] = d.t - peak_t; F[f'{s}_below_peak'] = cm - x
        if rel:
            r10 = pd.Series((x.values - _lag(d, s, 10)) / 10); g2 = r10.groupby(d.e)
            mu = g2.transform(lambda v: v.shift(1).expanding(5).mean()); sd = g2.transform(lambda v: v.shift(1).expanding(5).std())
            z = (r10 - mu) / (sd + 1e-6); F[f'{s}_z10'] = z; F[f'{s}_z10_cmax'] = z.groupby(d.e).cummax()
    X = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan).astype('float32'); X.index = d.sid.values
    return X.loc[frame.sample_id.astype(str).values]


def _build_compact(frame):
    d = pd.DataFrame({'sid': frame.sample_id.astype(str).values, 'e': frame.experiment_id.astype(str).values,
                      't': frame.relative_time_s.astype(float).values})
    for c in ['meas_t1', 'meas_t2', 'thermal_t', 'meas_p', 'meas_v', 'indu_t'] + META:
        d[c] = pd.to_numeric(frame[c], errors='coerce').values
    d = d.sort_values(['e', 't']).reset_index(drop=True)
    g = d.groupby('e', sort=False)
    # 시작 기준값: 실험 시작 30초(현재까지) 평균
    first30 = lambda c: d[c].where(d.t <= 30).groupby(d.e).transform(lambda s: s.expanding().mean()).groupby(d.e).ffill()
    d['tmax'] = d[['meas_t1', 'meas_t2']].max(1)
    v0 = first30('meas_v'); d['vrel'] = d.meas_v / v0.where(v0.abs() > 1e-3)
    F = {'t': d.t}
    for m in META: F[m] = d[m]
    for s in ['meas_t1', 'meas_t2', 'tmax', 'thermal_t', 'meas_p', 'vrel', 'indu_t']:
        x = d[s]; cm = g[s].cummax() if s != 'vrel' else d.groupby('e').vrel.cummax()
        F[s] = x; F[f'{s}_cmax'] = cm
        F[f'{s}_rise'] = x - first30(s) if s != 'vrel' else x - 1
        for w in (10, 30, 60):
            F[f'{s}_r{w}'] = (x.values - _lag(d, s, w)) / w
        F[f'{s}_r30_cmax'] = pd.Series(F[f'{s}_r30']).groupby(d.e).cummax()
        d['_ts'] = pd.to_datetime(d.t, unit='s')
        F[f'{s}_m60'] = d.groupby('e', sort=False).rolling('60s', on='_ts')[s].mean().reset_index(drop=True)
    F['vrel_cmin'] = d.groupby('e').vrel.cummin()
    vmax = d.groupby('e').meas_v.cummax(); F['v_over_peak'] = d.meas_v / vmax.where(vmax > 1e-3)
    F['v_peak_ratio'] = vmax / v0.where(v0.abs() > 1e-3)

    def since(mask): return d.t - d.t.where(mask).groupby(d.e).cummin()
    t1c = g.meas_t1.cummax(); thc = g.thermal_t.cummax(); vmin = F['vrel_cmin']
    vop_min = pd.Series(F['v_over_peak']).groupby(d.e).cummin()
    for th in (45, 50, 60, 80, 100, 150, 200, 300): F[f'since_t1_{th}'] = since(t1c >= th)
    for th in (0.8, 0.5, 0.1): F[f'since_v_{th}'] = since(vmin <= th)
    F['since_vpeak_0.8'] = since(vop_min <= 0.8)
    for th in (50, 100, 200, 300): F[f'since_th_{th}'] = since(thc >= th)
    for th in (5, 20): F[f'since_p_{th}'] = since(pd.Series(F['meas_p_rise']).groupby(d.e).cummax() >= th)
    X = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)
    X.index = d.sid.values
    return X.loc[frame.sample_id.astype(str).values].astype({c: 'float32' for c in X.columns if c not in META})
