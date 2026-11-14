"""Causal-only levers (no future info at any time step).

A. Learning curve  : CV with 50/75/100% of each fold's training experiments → is more data (Train+Val) worth it?
B. Seed ensemble   : 5 seeds on full Train → Val per seed vs average
C. Two-stage causal: stage-1 probs → causal onset features (cummax, time since P(≥k) crossed 0.3/0.5,
                     slope over 30/60s, EMA) → stage-2 GBM on [735 + onset].
                     Train rows use stage-1 OOF; Val/Test use the average of the same 5 fold models
                     (same distribution as OOF — avoids the stacking shift seen before).
Eval: GroupKFold(5, seed 0), Test-like 500-row cap, confidence cummax tau 0.7.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from exp_img import F, lgbm, load, log

LAB = F.LABELS
D = Path('/home/nxcolelxu/unid/data/battery_data_light')


def score(frame, P):
    c = F.cap_rows(frame).index.to_numpy()
    fc = frame.iloc[c].reset_index(drop=True)
    y = fc.label.to_numpy(int)
    lab = F.postprocess(fc, P[c], 0.7)
    return float(f1_score(y, lab, labels=LAB, average='macro')), f1_score(y, lab, labels=LAB, average=None).round(3).tolist()


def onset_feats(frame, P):
    """Strictly causal features from stage-1 class probs (per experiment, time-ordered)."""
    t_all = frame.relative_time_s.to_numpy(float)
    C = np.cumsum(P[:, ::-1], 1)[:, ::-1][:, 1:]          # P(≥k), k=2..6
    n = len(frame)
    out = {}
    def put(name, idx, v):
        out.setdefault(name, np.full(n, np.nan, np.float32))[idx] = v
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        t = t_all[idx]
        for j in range(5):
            c = C[idx, j]
            k = j + 2
            put(f'o_ge{k}', idx, c)
            put(f'o_ge{k}_cmax', idx, np.maximum.accumulate(c))
            for th in (0.3, 0.5):
                hit = c >= th
                first = np.where(hit.any(), np.argmax(hit), len(c))
                first_t = np.full(len(c), np.nan)
                # time since FIRST crossing up to now (causal: only defined after crossing)
                fc = np.maximum.accumulate(np.where(hit, np.arange(len(c)), -1))  # last crossing index so far
                firsts = np.where(np.maximum.accumulate(hit), t - t[min(first, len(c) - 1)], np.nan)
                put(f'o_ge{k}_since{int(th*10)}', idx, firsts)
                # fraction of last 60s above threshold
                cs = np.concatenate([[0], np.cumsum(hit)])
                lo = np.searchsorted(t, t - 60, side='left')
                r = np.arange(len(t))
                put(f'o_ge{k}_frac{int(th*10)}_60', idx, (cs[r + 1] - cs[lo]) / (r + 1 - lo))
            for w in (30, 60):
                jb = np.clip(np.searchsorted(t, t - w, side='right') - 1, 0, None)
                put(f'o_ge{k}_d{w}', idx, c - c[jb])
            ema = np.empty(len(c)); a = 0.2
            for r in range(len(c)):
                ema[r] = c[r] if r == 0 else a * c[r] + (1 - a) * ema[r - 1]
            put(f'o_ge{k}_ema', idx, ema)
    X = pd.DataFrame(out)
    X.index = frame.sample_id.astype(str).values
    return X


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    Xtr, Xva, Xte = F.build_features(tr), F.build_features(va), F.build_features(te)
    ytr = tr.label.astype(int)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    res = {}

    # A. learning curve
    rng = np.random.default_rng(0)
    for frac in (0.5, 0.75, 1.0):
        sc = []
        for rep in range(2 if frac < 1 else 1):
            oof = np.zeros((len(tr), 6))
            for a, b in splits:
                exps = tr.experiment_id.iloc[a].unique()
                keep = rng.choice(exps, max(2, int(round(len(exps) * frac))), replace=False)
                aa = a[tr.experiment_id.iloc[a].isin(keep).to_numpy()]
                oof[b] = lgbm(rep).fit(Xtr.iloc[aa], ytr.iloc[aa]).predict_proba(Xtr.iloc[b])
            sc.append(score(tr, oof)[0])
        res[f'A/frac{frac}'] = float(np.mean(sc))
        log(f'A learning curve frac {frac}: CV {np.mean(sc):.4f} ({len(sc)} reps)')

    # B. seed ensemble on Val
    pv = []
    for s in range(5):
        pv.append(lgbm(s).fit(Xtr, ytr).predict_proba(Xva))
        log(f'B seed {s}: Val {score(va, pv[-1])[0]:.4f}')
    res['B/val_per_seed'] = [score(va, p)[0] for p in pv]
    res['B/val_avg5'] = score(va, np.mean(pv, 0))[0]
    log(f'B 5-seed avg: Val {res["B/val_avg5"]:.4f}  (single mean {np.mean(res["B/val_per_seed"]):.4f}, std {np.std(res["B/val_per_seed"]):.4f})')

    # C. two-stage causal
    oof1 = np.zeros((len(tr), 6)); pva1 = np.zeros((len(va), 6)); pte1 = np.zeros((len(te), 6))
    for a, b in splits:
        m = lgbm(0).fit(Xtr.iloc[a], ytr.iloc[a])
        oof1[b] = m.predict_proba(Xtr.iloc[b])
        pva1 += m.predict_proba(Xva) / 5
        pte1 += m.predict_proba(Xte) / 5
    res['C/stage1_cv'] = score(tr, oof1)
    res['C/stage1_val_foldavg'] = score(va, pva1)
    log(f'C stage1: CV {res["C/stage1_cv"][0]:.4f} | Val (5-fold avg) {res["C/stage1_val_foldavg"][0]:.4f}')
    Otr, Ova, Ote = onset_feats(tr, oof1), onset_feats(va, pva1), onset_feats(te, pte1)
    X2tr, X2va, X2te = (pd.concat([X, O.loc[X.index]], axis=1) for X, O in ((Xtr, Otr), (Xva, Ova), (Xte, Ote)))
    for name, (A_, V_, T_) in {'onset_only': (Otr, Ova, Ote), 'base+onset': (X2tr, X2va, X2te)}.items():
        oof2 = np.zeros((len(tr), 6))
        for a, b in splits:
            oof2[b] = lgbm(0).fit(A_.iloc[a], ytr.iloc[a]).predict_proba(A_.iloc[b])
        pv2 = np.mean([lgbm(s).fit(A_, ytr).predict_proba(V_) for s in range(3)], 0)
        res[f'C/{name}'] = dict(cv=score(tr, oof2), val=score(va, pv2))
        log(f'C stage2 {name}: CV {res[f"C/{name}"]["cv"][0]:.4f} {res[f"C/{name}"]["cv"][1]} | '
            f'Val {res[f"C/{name}"]["val"][0]:.4f} {res[f"C/{name}"]["val"][1]}')
        np.save(f'causal_{name}_val.npy', pv2)
    Path('res_causal.json').write_text(json.dumps(res, indent=1))
    log('saved res_causal.json')


if __name__ == '__main__':
    main()
