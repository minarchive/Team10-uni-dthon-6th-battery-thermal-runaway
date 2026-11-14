"""여러 설정의 확률을 한 번에 저장 (디코더·앙상블 실험용).

저장: bank_<cfg>.npz
- oof_s{seed}_ids / oof_s{seed}_P : Train GroupKFold(5, seed) 검증 fold(500행 캡) 행의 OOF 확률
- val_ids / val_P   : Train 전체로 학습 → Val(500행 캡) 확률
- test_ids / test_P : Train 전체 학습 → Test 확률
- test_tv_P         : Train+Val 학습 → Test 확률

    python oof_bank.py --data_dir <D> --cfg v3+cos --emb emb_dinov2s.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402
from exp_img import fit_pca, img_features, lgbm, load, log  # noqa: E402

CFGS = {
    'v3': dict(),
    'v3+cos': dict(img=True, pca=False, cos=True),
    'v3+img_k32': dict(img=True, k=32),
}


def make_X(frame, emb, pca, cfg, cache):
    key = tuple(frame.sample_id.values[:3]) + (len(frame),)
    if key not in cache:
        cache.clear()
        cache[key] = F.build_features(frame)
    parts = [cache[key]]
    if cfg.get('img'):
        parts.append(img_features(frame, emb, pca, cfg.get('k', 32), use_pca=cfg.get('pca', True), use_cos=cfg.get('cos', True)))
    return pd.concat(parts, axis=1).loc[frame.sample_id.values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--cfg', required=True, choices=list(CFGS))
    ap.add_argument('--emb', type=Path, default=Path('emb_dinov2s.npz'))
    ap.add_argument('--seeds', default='0,1,2')
    args = ap.parse_args()
    cfg = CFGS[args.cfg]
    z = np.load(args.emb)
    emb = pd.DataFrame(np.concatenate([z['cls'], z['patch_mean']], 1).astype(np.float32), index=z['sample_id'].astype(str))
    tr, va, te = load(args.data_dir, 'train'), load(args.data_dir, 'validation'), load(args.data_dir, 'test')
    out = {}
    cache = {}
    need_pca = cfg.get('img') and cfg.get('pca', True)
    for s in map(int, args.seeds.split(',')):
        t0 = time.time()
        ids, Ps = [], []
        for a, b in GroupKFold(5, shuffle=True, random_state=s).split(tr, groups=tr.experiment_id):
            ftr, fev = tr.iloc[a], F.cap_rows(tr.iloc[b])
            pca = fit_pca(ftr, emb, cfg.get('k', 32)) if need_pca else None
            m = lgbm(s).fit(make_X(ftr, emb, pca, cfg, {}), ftr.label.astype(int))
            Ps.append(m.predict_proba(make_X(fev, emb, pca, cfg, {})))
            ids.append(fev.sample_id.to_numpy().astype(str))
        out[f'oof_s{s}_ids'], out[f'oof_s{s}_P'] = np.concatenate(ids), np.concatenate(Ps).astype(np.float32)
        log(f'{args.cfg} seed {s} OOF done ({time.time()-t0:.0f}s)')
    vc = F.cap_rows(va)
    pca = fit_pca(tr, emb, cfg.get('k', 32)) if need_pca else None
    m = lgbm(42).fit(make_X(tr, emb, pca, cfg, {}), tr.label.astype(int))
    out['val_ids'], out['val_P'] = vc.sample_id.to_numpy().astype(str), m.predict_proba(make_X(vc, emb, pca, cfg, {})).astype(np.float32)
    out['test_ids'], out['test_P'] = te.sample_id.to_numpy().astype(str), m.predict_proba(make_X(te, emb, pca, cfg, {})).astype(np.float32)
    tv = pd.concat([tr, va], ignore_index=True)
    pca = fit_pca(tv, emb, cfg.get('k', 32)) if need_pca else None
    m = lgbm(42).fit(make_X(tv, emb, pca, cfg, {}), tv.label.astype(int))
    out['test_tv_P'] = m.predict_proba(make_X(te, emb, pca, cfg, {})).astype(np.float32)
    np.savez(f'bank_{args.cfg}.npz', **out)
    log(f'saved bank_{args.cfg}.npz')


if __name__ == '__main__':
    main()
