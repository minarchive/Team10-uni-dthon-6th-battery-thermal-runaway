"""v3 (735) vs v3 + v4 experiment-relative onset features. Causal only.

CV: GroupKFold(5, seed 0) × GBM seeds {0,1,2}, 500-row cap, confidence cummax tau 0.7.
Val: full-Train fit, 3-seed average, scored once; experiment-level bootstrap Δ vs v3.
Saves v4_probs.npz (OOF/Val/Test for both sets).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import features_v4 as V
import moe_gate as M
from exp_img import F, lgbm, load, log

LAB = F.LABELS
D = Path('/home/nxcolelxu/unid/data/battery_data_light')
SEEDS = (0, 1, 2)


def labels(frame, P):
    c = F.cap_rows(frame).index.to_numpy()
    fc = frame.iloc[c].reset_index(drop=True)
    return fc, F.postprocess(fc, P[c], 0.7)


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    import re, sys
    GROUPS = {
        'p': lambda c: c.startswith('z_meas_p'),                                   # pressure onset only (2→3 hypothesis)
        'volt': lambda c: c.startswith('v_since_l1frac') or c.startswith('v_since_drop') or c == 'v_drop_from_max',
        'hot': lambda c: re.match(r'(meas_t1|t_max)_since(350|400|500)$', c) is not None,
    }
    def make(groups):
        def fn(frame):
            X = F.build_features(frame)
            if not groups:
                return X
            Z = V.build_v4(frame).loc[X.index]
            keep = [c for c in Z.columns if any(GROUPS[g](c) for g in groups)]
            return pd.concat([X, Z[keep]], axis=1)
        return fn
    combos = [a.split('+') if a != 'v3' else [] for a in (sys.argv[1:] or ['v3', 'p', 'volt', 'hot', 'p+hot'])]
    sets = {('v3' if not g else 'v3+' + '+'.join(g)): make(g) for g in combos}
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    res, save, lab_cache = {}, {}, {}
    for name, fn in sets.items():
        Xtr, Xva, Xte = fn(tr), fn(va), fn(te)
        ytr = tr.label.astype(int)
        cv_scores, oof_avg = [], np.zeros((len(tr), 6))
        for s in SEEDS:
            oof = np.zeros((len(tr), 6))
            for a, b in splits:
                oof[b] = lgbm(s).fit(Xtr.iloc[a], ytr.iloc[a]).predict_proba(Xtr.iloc[b])
            fc, l = labels(tr, oof)
            cv_scores.append(f1_score(fc.label.to_numpy(int), l, labels=LAB, average='macro'))
            oof_avg += oof / len(SEEDS)
        pv = np.mean([lgbm(s).fit(Xtr, ytr).predict_proba(Xva) for s in SEEDS], 0)
        pt = np.mean([lgbm(s).fit(Xtr, ytr).predict_proba(Xte) for s in SEEDS], 0)
        fcv, lv = labels(va, pv)
        fct, lt = labels(tr, oof_avg)
        lab_cache[name] = (fct, lt, fcv, lv)
        res[name] = dict(n_feat=Xtr.shape[1], cv_per_seed=cv_scores, cv_mean=float(np.mean(cv_scores)), cv_std=float(np.std(cv_scores)),
                         cv_avg3_per_class=f1_score(fct.label.to_numpy(int), lt, labels=LAB, average=None).round(3).tolist(),
                         val=float(f1_score(fcv.label.to_numpy(int), lv, labels=LAB, average='macro')),
                         val_per_class=f1_score(fcv.label.to_numpy(int), lv, labels=LAB, average=None).round(3).tolist())
        log(f'{name:6s} feats {Xtr.shape[1]}  CV {res[name]["cv_mean"]:.4f} ± {res[name]["cv_std"]:.4f} {res[name]["cv_avg3_per_class"]} | '
            f'Val {res[name]["val"]:.4f} {res[name]["val_per_class"]}')
        save.update({f'{name}_oof': oof_avg, f'{name}_val': pv, f'{name}_test': pt})
        if False:
            m = lgbm(0).fit(Xtr, ytr)
            imp = pd.Series(getattr(m, 'feature_importances_', np.zeros(Xtr.shape[1])), index=Xtr.columns)
            v4cols = [c for c in Xtr.columns if c not in F.build_features(va.head(5)).columns]
            res['v4_importance_share'] = float(imp[v4cols].sum() / max(imp.sum(), 1e-9))
            res['top_v4'] = imp[v4cols].sort_values(ascending=False).head(15).round(1).to_dict()
            log(f'v4 share of importance {res["v4_importance_share"]:.3f}; top v4: {list(res["top_v4"])[:8]}')
    fct, lt0, fcv, lv0 = lab_cache['v3']
    for name in sets:
        if name == 'v3':
            continue
        _, lt1, _, lv1 = lab_cache[name]
        cb = M.boot_diff(fct, lt0, lt1, np.arange(len(fct)), n=1000)
        vb = M.boot_diff(fcv, lv0, lv1, np.arange(len(fcv)), n=1000)
        res[name]['cv_boot'], res[name]['val_boot'] = cb, vb
        log(f'Δ({name} − v3)  CV {cb[0]:+.4f} [{cb[1]:+.4f},{cb[2]:+.4f}]  Val {vb[0]:+.4f} [{vb[1]:+.4f},{vb[2]:+.4f}] P(Δ>0)={vb[3]:.2f}')
    Path('res_v4_ablate.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez('v4_ablate_probs.npz', train_id=tr.sample_id.to_numpy(str), val_id=va.sample_id.to_numpy(str),
             test_id=te.sample_id.to_numpy(str), **save)
    log('saved res_v4.json, v4_probs.npz')


if __name__ == '__main__':
    main()
