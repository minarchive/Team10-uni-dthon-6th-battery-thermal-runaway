"""절대값 vs 상대값 피처 ablation — "Test 분포가 다르면 절대 수준 피처가 해롭지 않은가"를 확인.

변형
- v3            : 기준 735피처
- v3-atmp       : atmp_t 관련 피처 제거 (Test는 atmp_t 가 한 번도 50°C를 넘지 않음)
- rel           : 절대 수준 피처(원값, cummax/cummin, 윈도우 mean/max/min) 제거,
                  대신 시작값 대비 차이(…_rel = 값 − 시작값) 로 대체. 변화량·기울기·경과시간·임계도달 피처는 유지
- rel-atmp      : rel 에서 atmp_t 제거
- rel-cross     : rel 에서 임계 도달(절대 온도 45/50/…°C) 피처까지 제거 → 완전 상대
- v3/bat, rel/bat : bat_class 별로 따로 학습 (각 26개 실험)

평가: Train GroupKFold(5)×3 seeds (500행 캡, 확신 cummax τ=0.7) + Val.
    python exp_relative.py --data_dir <battery_data_light>
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402
from exp_img import lgbm, load, log, score6  # noqa: E402

ABS_PAT = re.compile(r'^(?P<s>.+?)(_cummax|_cummin|_mean\d+|_max\d+|_min\d+)?$')


def to_relative(X):
    """절대 수준 컬럼을 시작값 대비 차이로 바꿉니다."""
    X = X.copy()
    out = {}
    drop = []
    for s in F.SIGNALS:
        start = X[s] - X[f'{s}_from_start']
        drop.append(s)
        for suf in ['_cummax', '_cummin'] + [f'_{a}{w}' for w in F.WINDOWS_S for a in ['mean', 'max', 'min']]:
            c = f'{s}{suf}'
            out[f'{c}_rel'] = X[c] - start
            drop.append(c)
    X = X.drop(columns=drop)
    return pd.concat([X, pd.DataFrame(out, index=X.index)], axis=1)


def variant(X, name):
    if name.startswith('rel'):
        X = to_relative(X)
    if 'atmp' in name:
        X = X[[c for c in X.columns if not c.startswith('atmp_t')]]
    if 'cross' in name:
        cross = [c for c in X.columns if re.search(r'_since\d+$|_since45$|^v_since_below|^v_ratio_cummin$', c)]
        X = X.drop(columns=cross)
    return X


def fit_predict(Xtr, ytr, Xev, btr, bev, per_bat, seed):
    if not per_bat:
        return lgbm(seed).fit(Xtr, ytr).predict_proba(Xev)
    P = np.zeros((len(Xev), 6))
    for b in np.unique(bev):
        m = lgbm(seed).fit(Xtr[btr == b], ytr[btr == b])
        pb = m.predict_proba(Xev[bev == b])
        P[np.where(bev == b)[0][:, None], m.classes_ - 1] = pb
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('res_relative.json'))
    args = ap.parse_args()
    tr, va = load(args.data_dir, 'train'), load(args.data_dir, 'validation')
    Xtr_full, Xva_full = F.build_features(tr), F.build_features(F.cap_rows(va))
    vc = F.cap_rows(va).reset_index(drop=True)
    names = ['v3', 'v3-atmp', 'rel', 'rel-atmp', 'rel-atmp-cross', 'v3/bat', 'rel-atmp/bat']
    res = {}
    for name in names:
        t0 = time.time()
        base = name.split('/')[0]
        per_bat = name.endswith('/bat')
        Xa = variant(Xtr_full, base)
        rows = []
        for s in [0, 1, 2]:
            frames, ps = [], []
            for a, b in GroupKFold(5, shuffle=True, random_state=s).split(tr, groups=tr.experiment_id):
                ftr = tr.iloc[a]
                fev = F.cap_rows(tr.iloc[b])
                Xtr, Xev = Xa.loc[ftr.sample_id.values], Xa.loc[fev.sample_id.values]
                P = fit_predict(Xtr, ftr.label.astype(int).to_numpy(), Xev,
                                ftr.bat_class.to_numpy(), fev.bat_class.to_numpy(), per_bat, s)
                frames.append(fev); ps.append(P)
            rows.append(score6(pd.concat(frames), np.concatenate(ps)))
        r = {k: (float(np.mean([x[k] for x in rows])), float(np.std([x[k] for x in rows]))) for k in ['raw', 'post']}
        r['per_class_post'] = np.mean([x['per_class'] for x in rows], 0).round(4).tolist()
        Xv = variant(Xva_full, base)[Xa.columns]
        Pv = fit_predict(Xa, tr.label.astype(int).to_numpy(), Xv, tr.bat_class.to_numpy(), vc.bat_class.to_numpy(), per_bat, 42)
        rv = score6(vc, Pv)
        r['val'] = dict(raw=rv['raw'], post=rv['post'], per_class=rv['per_class'])
        r['n_features'] = int(Xa.shape[1])
        res[name] = r
        log(f'{name:16s} feats {Xa.shape[1]:4d} | CV raw {r["raw"][0]:.4f}±{r["raw"][1]:.4f} post {r["post"][0]:.4f}±{r["post"][1]:.4f} '
            f'| VAL raw {rv["raw"]:.4f} post {rv["post"]:.4f} | F1/cls(CV) {r["per_class_post"]} ({time.time()-t0:.0f}s)')
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
