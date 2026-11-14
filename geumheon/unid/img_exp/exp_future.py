"""Does seeing the future (within the same experiment) fix transition lag?

Feature sets (same GBM, same GroupKFold(5, seed 0) folds, Test-like 500-row cap for eval):
  past   : val0.9195 build_features (735, causal)
  both   : past + the same builder run on the time-reversed experiment (future-window mirror, 'fut_' prefix)
Decoding:
  cummax : causal confidence cummax (tau 0.7), current pipeline
  viterbi: offline best non-decreasing label path, sum log p

Writes res_future.json and future_probs.npz (OOF / Val / Test probs for both feature sets).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from exp_img import F, lgbm, load, log

LAB = F.LABELS
D = Path('/home/nxcolelxu/unid/data/battery_data_light')


def reversed_frame(frame):
    f = frame.copy()
    t = f.relative_time_s.astype(float)
    f['relative_time_s'] = f.groupby('experiment_id').relative_time_s.transform(lambda s: s.astype(float).max()) - t
    return f


def feats(frame, use_future):
    X = F.build_features(frame)
    if not use_future:
        return X
    if use_future == 'look':
        return pd.concat([X, lookahead(frame).loc[X.index]], axis=1)
    Xr = F.build_features(reversed_frame(frame)).loc[X.index]
    Xr.columns = [f'fut_{c}' for c in Xr.columns]
    return pd.concat([X, Xr], axis=1)


LOOK = (30, 60, 120)


def lookahead(frame):
    """Bounded future: for each signal, mean over (t, t+w] minus value at t, and value at t+w minus value at t.
    No absolute time and no time-to-end. Near the recording end the window is truncated (same in all splits)."""
    d = F._prep(frame)
    n = len(d)
    t_all = d.relative_time_s.to_numpy(float)
    cols = {}
    for _, idx in d.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        t = t_all[idx]
        for sname in F.SIGNALS:
            v = d[sname].to_numpy(float)[idx]
            v = pd.Series(v).ffill().bfill().to_numpy()
            if np.isnan(v).all():
                continue
            cs = np.concatenate([[0], np.cumsum(v)])
            for w in LOOK:
                hi = np.searchsorted(t, t + w, side='right')            # rows in (t, t+w] are [r+1, hi)
                r = np.arange(len(t))
                cnt = np.maximum(hi - r - 1, 1)
                fmean = np.where(hi - r - 1 > 0, (cs[hi] - cs[r + 1]) / cnt, v)
                fend = v[np.maximum(hi - 1, r)]
                for nm, val in ((f'la_{sname}_mean{w}', fmean - v), (f'la_{sname}_end{w}', fend - v)):
                    if nm not in cols:
                        cols[nm] = np.full(n, np.nan, np.float32)
                    cols[nm][idx] = val
    X = pd.DataFrame(cols)
    X.index = d.sample_id.values
    return X.loc[frame.sample_id.astype(str).values]


def viterbi_mono(P):
    L = np.log(np.clip(P, 1e-6, 1))
    n = len(L)
    Dp = L[0].copy()
    B = np.zeros((n, 6), int)
    for t in range(1, n):
        arg = np.zeros(6, int)
        a = 0
        for k in range(6):
            if Dp[k] >= Dp[a]:
                a = k
            arg[k] = a
        Dp = Dp[arg] + L[t]
        B[t] = arg
    y = np.zeros(n, int)
    y[-1] = Dp.argmax()
    for t in range(n - 1, 0, -1):
        y[t - 1] = B[t, y[t]]
    return y + 1


def decode(frame, P, mode):
    out = np.empty(len(frame), int)
    t = frame.relative_time_s.to_numpy(float)
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        out[idx] = F.conf_cummax(P[idx], 0.7) if mode == 'cummax' else viterbi_mono(P[idx])
    return out


def score(frame, P, mode):
    """Test-like: cap rows, then features were computed on full frame (same as existing pipeline)."""
    c = F.cap_rows(frame).index.to_numpy()
    fc = frame.iloc[c].reset_index(drop=True)
    y = fc.label.to_numpy(int)
    lab = decode(fc, P[c], mode)
    return float(f1_score(y, lab, labels=LAB, average='macro')), f1_score(y, lab, labels=LAB, average=None).round(3).tolist()


import sys
VARIANTS = [('past', False), ('both', True)] if len(sys.argv) < 2 else [(v, {'past': False, 'both': True, 'look': 'look'}[v]) for v in sys.argv[1:]]


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    res, save = {}, {}
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    for name, fut in VARIANTS:
        t0 = time.time()
        Xtr, Xva, Xte = feats(tr, fut), feats(va, fut), feats(te, fut)
        log(f'{name}: features {Xtr.shape} in {time.time()-t0:.0f}s')
        oof = np.zeros((len(tr), 6))
        for a, b in splits:
            m = lgbm(0).fit(Xtr.iloc[a], tr.label.iloc[a].astype(int))
            oof[b] = m.predict_proba(Xtr.iloc[b])
        m = lgbm(42).fit(Xtr, tr.label.astype(int))
        pva, pte = m.predict_proba(Xva), m.predict_proba(Xte)
        save.update({f'{name}_oof': oof, f'{name}_val': pva, f'{name}_test': pte})
        for mode in ('cummax', 'viterbi'):
            cv, cvp = score(tr, oof, mode)
            vv, vvp = score(va, pva, mode)
            res[f'{name}/{mode}'] = dict(cv=cv, cv_per_class=cvp, val=vv, val_per_class=vvp)
            log(f'{name:5s} {mode:8s} CV {cv:.4f} {cvp} | Val {vv:.4f} {vvp}')
    Path(f"res_future_{'_'.join(n for n, _ in VARIANTS)}.json").write_text(json.dumps(res, indent=1))
    np.savez(f"future_probs_{'_'.join(n for n, _ in VARIANTS)}.npz", train_id=tr.sample_id.to_numpy(str), val_id=va.sample_id.to_numpy(str),
             test_id=te.sample_id.to_numpy(str), **save)
    log('saved res_future.json, future_probs.npz')


if __name__ == '__main__':
    main()
