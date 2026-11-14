"""v4 causal features: experiment-relative onsets (adds to val0.9195 build_features, 735 → ~935).

Motivation (Train+Val label analysis, 97 experiments):
  - 2→3 has no absolute sensor rule, but pressure relative to the experiment's own L1 baseline
    (first > baseline + 5σ after L2 starts) fires at median +15 s from L3 start, 43% within ±60 s
    (best absolute threshold: 13%). Image embeddings do not mark L3 (≤10%).
  - 3→4 = voltage drop starts (current rules start at 10% drop) → finer relative drops.
  - 5→6 = t1 very high (median 388 °C; current levels stop at 300) → 350/400/500 °C.

Causality: L1 baseline = rows so far with cummax(t1) < 45 °C (the 1→2 rule, known when it happens).
Every feature at time t uses rows with time ≤ t of the same experiment only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402

Z_LEVELS = (5, 10, 20)
Z_WINDOWS = (30, 60)
V_DROP_FRAC = (0.99, 0.98, 0.95)
V_DROP_ABS = (0.05, 0.1, 0.2, 0.5)
HOT_LEVELS = (350, 400, 500)


def _expanding_masked_stats(x, m):
    """Running mean/std of x over rows where m is True (rows so far). NaN until first such row."""
    xv = np.where(m & ~np.isnan(x), x, 0.0)
    c = np.cumsum(m & ~np.isnan(x))
    s1, s2 = np.cumsum(xv), np.cumsum(xv * xv)
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.where(c > 0, s1 / np.maximum(c, 1), np.nan)
        var = np.where(c > 1, s2 / np.maximum(c, 1) - mu * mu, np.nan)
    return mu, np.sqrt(np.clip(var, 0, None))


def _since_first(t, hit):
    """time since first True so far (NaN before)."""
    seen = np.maximum.accumulate(hit)
    first = np.argmax(hit) if hit.any() else 0
    return np.where(seen, t - t[first], np.nan)


def _win_max(t, x, w):
    out = np.empty(len(x))
    lo = np.searchsorted(t, t - w, side='left')
    for r in range(len(x)):          # short windows, small n per experiment
        seg = x[lo[r]:r + 1]
        out[r] = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
    return out


def build_v4(frame: pd.DataFrame) -> pd.DataFrame:
    d = F._prep(frame)
    n = len(d)
    t_all = d.relative_time_s.to_numpy(float)
    cols = {}

    def put(name, idx, v):
        cols.setdefault(name, np.full(n, np.nan, np.float32))[idx] = v

    for _, idx in d.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        t = t_all[idx]
        t1 = d['meas_t1'].to_numpy(float)[idx]
        cm1 = np.fmax.accumulate(np.nan_to_num(t1, nan=-np.inf))
        in_l1 = cm1 < 45
        after_l2 = ~in_l1
        for s in F.SIGNALS:
            x = d[s].to_numpy(float)[idx]
            if np.isnan(x).all():
                continue
            mu, sd = _expanding_masked_stats(x, in_l1)
            mu = pd.Series(mu).ffill().to_numpy()       # baseline frozen once L2 starts
            sd = pd.Series(sd).ffill().to_numpy()
            sd = np.fmax(sd, 1e-3 * np.fmax(np.abs(mu), 1.0))
            z = np.clip((x - mu) / sd, -1e4, 1e4)        # flat baselines → huge z; clip for float32
            zc = np.fmax.accumulate(np.nan_to_num(np.where(after_l2, z, -np.inf), nan=-np.inf))
            zc = np.where(np.isfinite(zc), zc, np.nan)
            put(f'z_{s}', idx, z)
            put(f'z_{s}_cmax_l2', idx, zc)
            for k in Z_LEVELS:
                put(f'z_{s}_since{k}', idx, _since_first(t, after_l2 & (np.nan_to_num(z, nan=-np.inf) > k)))
            for w in Z_WINDOWS:
                put(f'z_{s}_max{w}', idx, _win_max(t, z, w))
        # fine voltage drop (relative to L1 baseline mean, and absolute from running max)
        v = d['meas_v'].to_numpy(float)[idx]
        vmu, _ = _expanding_masked_stats(v, in_l1)
        vmu = pd.Series(vmu).ffill().to_numpy()
        vmax = np.fmax.accumulate(np.nan_to_num(v, nan=-np.inf))
        vmin = np.fmin.accumulate(np.nan_to_num(v, nan=np.inf))
        for f in V_DROP_FRAC:
            put(f'v_since_l1frac{int(f * 100)}', idx, _since_first(t, (vmin <= f * vmu) & (vmu > 0)))
        for a in V_DROP_ABS:
            put(f'v_since_drop{a}', idx, _since_first(t, (vmax - vmin) >= a))
        put('v_drop_from_max', idx, vmax - v)
        # very high temperature
        for s in ('meas_t1', 't_max'):
            cm = np.fmax.accumulate(np.nan_to_num(d[s].to_numpy(float)[idx], nan=-np.inf))
            for lv in HOT_LEVELS:
                put(f'{s}_since{lv}', idx, _since_first(t, cm >= lv))
        put('since_l2', idx, _since_first(t, after_l2))
    X = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    X.index = d.sample_id.values
    return X.loc[frame.sample_id.astype(str).values]


def build_all(frame: pd.DataFrame) -> pd.DataFrame:
    X = F.build_features(frame)
    return pd.concat([X, build_v4(frame).loc[X.index]], axis=1)
