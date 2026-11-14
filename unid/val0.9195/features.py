"""배터리 열폭주 6단계 분류 — 시계열 피처 + 후처리 (val0.9195 모델용).

원칙
- 이미지는 사용하지 않습니다(CSV 센서 값만 사용).
- 모든 피처는 같은 experiment_id 안에서 relative_time_s <= 현재 시점인 관측만 사용합니다(인과적).
- 실험 전체 통계(전체 평균·최댓값·길이 등)는 사용하지 않습니다.
- 윈도우는 '행 수'가 아니라 '초' 기준입니다. Test의 긴 실험은 500행으로 줄어 있어 간격이 10/15초로 섞여 있기 때문입니다.
- 입력 CSV에 들어 있는 행만으로 계산하며, 실험끼리 독립입니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
LABELS = np.array([1, 2, 3, 4, 5, 6])

# meas_a(전부 0), meas_t1_plus_t2(t1+t2 중복), indu_t.1(결측 79%)은 사용하지 않음
GAS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf',
       'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
BASE_SIGNALS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'atmp_t', 'thermal_t', 'indu_t'] + GAS
DERIVED = ['t_max', 't_diff', 'gas_sum']
SIGNALS = BASE_SIGNALS + DERIVED
META = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']   # 실험마다 고정된 조건값(입력 컬럼)
WINDOWS_S = [30, 60, 120, 300]
TEMP_LEVELS = [45, 50, 60, 80, 100, 150, 200, 300]
V_FRACS = [0.9, 0.5, 0.1]
MAX_ROWS = 500


# ---------------------------------------------------------------------------
# 1. 피처
# ---------------------------------------------------------------------------
def _prep(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ['relative_time_s'] + BASE_SIGNALS + META
    d = frame[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    d['experiment_id'] = frame['experiment_id'].astype(str).values
    d['sample_id'] = frame['sample_id'].astype(str).values
    d['t_max'] = d[['meas_t1', 'meas_t2']].max(axis=1)
    d['t_diff'] = d['meas_t1'] - d['meas_t2']
    d['gas_sum'] = d[GAS].sum(axis=1, min_count=1)
    return d


def _crossing_features(d: pd.DataFrame) -> dict:
    """처음으로 임계값을 넘은 시각 이후 경과 시간(넘기 전 NaN). 모두 과거 관측만 사용.

    데이터 분석 근거
    - 1→2 전환은 Train/Val 100% 'meas_t1 누적최대 >= 45°C 최초 시점'
    - 3단계는 항상 meas_t1 >= 50°C 이후에 시작
    - 4·5단계는 전압 하락(→ 0V)과 함께 시작
    """
    out = {}
    eid, t = d['experiment_id'], d['relative_time_s']

    def since_first(mask, name):
        t_first = t.where(mask).groupby(eid).cummin()
        out[name] = t - t_first

    for s in ['meas_t1', 'meas_t2', 't_max']:
        cm = d[s].groupby(eid).cummax()
        for lv in TEMP_LEVELS:
            since_first(cm >= lv, f'{s}_since{lv}')
    cm1 = d['meas_t1'].groupby(eid).cummax()
    rise = cm1 - 45
    out['t1_rise_since45'] = rise.where(cm1 >= 45)
    out['t1_rate_since45'] = (rise / out['meas_t1_since45'].clip(lower=5)).where(cm1 >= 45)
    v0 = d['meas_v'].groupby(eid).transform('first')
    vr = (d['meas_v'] / v0.where(v0 > 0)).groupby(eid).cummin()
    out['v_ratio_cummin'] = vr
    for f in V_FRACS:
        since_first(vr <= f, f'v_since_below{int(f * 100)}')
    return out


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """피처 표(735개)를 입력 행 순서 그대로 반환합니다. 인덱스는 sample_id."""
    d = _prep(frame)
    order = np.lexsort((d['relative_time_s'].to_numpy(), d['experiment_id'].to_numpy()))
    d = d.iloc[order].reset_index(drop=True)
    d['_ts'] = pd.to_datetime(d['relative_time_s'], unit='s')
    g = d.groupby('experiment_id', sort=False)

    feats = {'relative_time_s': d['relative_time_s'], **{m: d[m] for m in META}}
    t = d['relative_time_s']

    # 처음부터 현재까지(누적) 피처
    for s in SIGNALS:
        x, gx = d[s], g[s]
        feats[s] = x
        cmax, cmin = gx.cummax(), gx.cummin()
        feats[f'{s}_cummax'] = cmax
        feats[f'{s}_cummin'] = cmin
        feats[f'{s}_from_start'] = x - gx.transform('first')
        feats[f'{s}_below_peak'] = cmax - x
        feats[f'{s}_above_min'] = x - cmin
        is_peak = x.eq(cmax) & x.notna()
        last_peak_t = t.where(is_peak).groupby(d['experiment_id']).ffill()
        feats[f'{s}_since_peak'] = t - last_peak_t

    # 시간(초) 기준 과거 윈도우 피처
    tmp = d[['experiment_id', '_ts', 'relative_time_s'] + SIGNALS].copy()
    for s in SIGNALS:
        tmp[f'{s}__tx'] = tmp[s] * tmp['relative_time_s']
    tmp['tt'] = tmp['relative_time_s'] ** 2
    for w in WINDOWS_S:
        r = tmp.groupby('experiment_id', sort=False).rolling(f'{w}s', on='_ts', closed='right')
        cols = SIGNALS + [f'{s}__tx' for s in SIGNALS] + ['relative_time_s', 'tt']
        mean = r[cols].mean().reset_index(drop=True)
        std = r[SIGNALS].std().reset_index(drop=True)
        mx = r[SIGNALS].max().reset_index(drop=True)
        mn = r[SIGNALS].min().reset_index(drop=True)
        cnt = r['relative_time_s'].count().reset_index(drop=True)
        var_t = mean['tt'] - mean['relative_time_s'] ** 2
        lagged = pd.merge_asof(   # w초 전(이하) 마지막 관측값
            pd.DataFrame({'_k': d['relative_time_s'] - w, 'experiment_id': d['experiment_id'],
                          '_i': np.arange(len(d))}).sort_values('_k'),
            d[['relative_time_s', 'experiment_id'] + SIGNALS].rename(columns={'relative_time_s': '_k'}).sort_values('_k'),
            on='_k', by='experiment_id', direction='backward',
        ).sort_values('_i').reset_index(drop=True)
        for s in SIGNALS:
            feats[f'{s}_mean{w}'] = mean[s]
            feats[f'{s}_std{w}'] = std[s]
            feats[f'{s}_max{w}'] = mx[s]
            feats[f'{s}_min{w}'] = mn[s]
            feats[f'{s}_dev{w}'] = d[s] - mean[s]
            cov = mean[f'{s}__tx'] - mean[s] * mean['relative_time_s']
            feats[f'{s}_slope{w}'] = (cov / var_t).where((cnt >= 3) & (var_t > 0))
            feats[f'{s}_delta{w}'] = (d[s] - lagged[s]) / w

    # 임계 도달 이후 경과 시간 피처
    feats.update(_crossing_features(d))

    X = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    X.index = d['sample_id'].values
    return X.loc[frame['sample_id'].astype(str).values]


# ---------------------------------------------------------------------------
# 2. 모델
# ---------------------------------------------------------------------------
def build_model():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
        class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1,
    )


# ---------------------------------------------------------------------------
# 3. 후처리: 확신 조건 cummax
# ---------------------------------------------------------------------------
def conf_cummax(P: np.ndarray, tau: float = 0.7) -> np.ndarray:
    """한 실험(시간순 정렬)의 확률 P(n×6)에 확신 조건 cummax를 적용합니다.

    - 확률 >= tau 인 예측만 '확정 단계'로 누적(누적 최댓값)
    - 출력 = max(확정 단계, 현재 예측)
    → 단계는 내려가지 않고(라벨이 실험 안에서 단조 증가함), 한 번 튄 저확신 오예측은 그 행에만 남고 이후로 전파되지 않음
    """
    pred, conf = LABELS[P.argmax(1)], P.max(1)
    committed = np.maximum.accumulate(np.where(conf >= tau, pred, 1))
    return np.maximum(committed, pred)


def postprocess(frame: pd.DataFrame, P: np.ndarray, tau: float = 0.7) -> np.ndarray:
    """실험별로 시간순 정렬해 conf_cummax 적용 후, 입력 행 순서로 라벨을 돌려줍니다."""
    out = np.empty(len(frame), dtype=int)
    t = pd.to_numeric(frame['relative_time_s']).to_numpy(float)
    for _, idx in frame.reset_index(drop=True).groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        out[idx] = conf_cummax(P[idx], tau)
    return out


# ---------------------------------------------------------------------------
# 4. 평가용: Test와 같은 조건 만들기
# ---------------------------------------------------------------------------
def cap_rows(frame: pd.DataFrame, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """max_rows를 넘는 실험을 시간순 균등 다운샘플링합니다(Test의 긴 실험이 500행으로 줄어 있는 것과 같은 방식)."""
    parts = []
    for _, e in frame.groupby('experiment_id', sort=False):
        e = e.sort_values('relative_time_s')
        if len(e) > max_rows:
            e = e.iloc[np.unique(np.round(np.linspace(0, len(e) - 1, max_rows)).astype(int))]
        parts.append(e)
    return pd.concat(parts).sort_index()
