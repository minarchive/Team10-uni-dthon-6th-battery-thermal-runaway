"""파인튜닝 모델의 OOF 확률을 시계열 피처(v3)에 스태킹 → LightGBM.

- Train 행의 확률은 finetune.py의 OOF(해당 실험을 보지 않은 fold 모델)입니다. in-sample 확률은 쓰지 않습니다.
- Val/Test 행은 5개 fold 모델 평균 확률입니다(Val은 어느 모델도 학습에 쓰지 않음 → 가장 깨끗한 확인 점수).
- CV의 fold는 finetune.py와 같은 GroupKFold(5, seed 0)를 기본으로 씁니다. 그래도 학습 fold 행의 OOF 확률을 만든
  모델이 평가 fold 실험을 봤을 수 있어(2차 누수) CV는 조금 낙관적일 수 있습니다. Val이 더 믿을 만합니다.

    python exp_stack_img.py --data_dir <battery_data_light> --probs ft_convnext_tiny.npz [--emb emb_dinov2s.npz]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402
from exp_img import fit_pca, img_features, lgbm, load, log, score6  # noqa: E402

WINDOWS = [30, 60]


def prob_features(frame, probs):
    P = probs.loc[frame.sample_id.values].to_numpy(np.float32)
    t_all = frame.relative_time_s.to_numpy(float)
    n = len(frame)
    cols = {f'cnn_p{k+1}': P[:, k] for k in range(6)}
    cols['cnn_stage'] = P @ np.arange(1, 7, dtype=np.float32)
    for name in [f'cnn_p{k+1}_cummax' for k in range(6)] + ['cnn_stage_cummax'] + \
                [f'cnn_p{k+1}_mean{w}' for k in range(6) for w in WINDOWS] + \
                [f'cnn_stage_mean{w}' for w in WINDOWS] + [f'cnn_stage_delta{w}' for w in WINDOWS]:
        cols[name] = np.full(n, np.nan, np.float32)
    for _, idx in frame.reset_index(drop=True).groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        t, p, st = t_all[idx], P[idx], cols['cnn_stage'][idx]
        for k in range(6):
            cols[f'cnn_p{k+1}_cummax'][idx] = np.maximum.accumulate(p[:, k])
        cols['cnn_stage_cummax'][idx] = np.maximum.accumulate(st)
        cp = np.vstack([np.zeros((1, 6)), np.cumsum(p, 0)])
        cs = np.r_[0, np.cumsum(st)]
        ar = np.arange(len(t))
        for w in WINDOWS:
            lo = np.searchsorted(t, t - w, side='right')
            lag = lo - 1
            cnt = ar + 1 - lo
            m = (cp[ar + 1] - cp[lo]) / cnt[:, None]
            for k in range(6):
                cols[f'cnn_p{k+1}_mean{w}'][idx] = m[:, k]
            cols[f'cnn_stage_mean{w}'][idx] = (cs[ar + 1] - cs[lo]) / cnt
            cols[f'cnn_stage_delta{w}'][idx] = np.where(lag >= 0, st - st[np.clip(lag, 0, None)], np.nan)
    return pd.DataFrame(cols, index=frame.sample_id.values)


def make_X(frame, cfg, probs, emb, pca):
    parts = []
    if cfg.get('ts', True):
        parts.append(F.build_features(frame))
    if cfg.get('cnn'):
        parts.append(prob_features(frame, probs))
    if cfg.get('img'):
        parts.append(img_features(frame, emb, pca, cfg.get('k', 32)))
    return pd.concat(parts, axis=1).loc[frame.sample_id.values]


def run_cv(tr, cfg, probs, emb, seeds):
    out = []
    for s in seeds:
        frames, ps = [], []
        for a, b in GroupKFold(5, shuffle=True, random_state=s).split(tr, groups=tr.experiment_id):
            ftr, fev = tr.iloc[a], F.cap_rows(tr.iloc[b])
            pca = fit_pca(ftr, emb, cfg.get('k', 32)) if cfg.get('img') else None
            m = lgbm(s).fit(make_X(ftr, cfg, probs, emb, pca), ftr.label.astype(int))
            frames.append(fev); ps.append(m.predict_proba(make_X(fev, cfg, probs, emb, pca)))
        out.append(score6(pd.concat(frames), np.concatenate(ps)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--probs', type=Path, required=True, nargs='+', help='finetune.py 출력 npz (여러 개면 평균)')
    ap.add_argument('--emb', type=Path, default=None, help='DINOv2 임베딩 npz (주면 img 피처 조합도 평가)')
    ap.add_argument('--out', type=Path, default=Path('res_stack.json'))
    args = ap.parse_args()

    zs = [np.load(p) for p in args.probs]
    ids = zs[0]['sample_id'].astype(str)
    probs = pd.DataFrame(np.mean([z['probs'] for z in zs], 0), index=ids)
    emb = None
    if args.emb:
        z = np.load(args.emb)
        emb = pd.DataFrame(np.concatenate([z['cls'], z['patch_mean']], 1).astype(np.float32), index=z['sample_id'].astype(str))
    tr, va = load(args.data_dir, 'train'), load(args.data_dir, 'validation')

    cfgs = {'v3': dict(ts=True), 'cnn_only': dict(ts=False, cnn=True), 'v3+cnn': dict(ts=True, cnn=True)}
    if emb is not None:
        cfgs['v3+cnn+img_k32'] = dict(ts=True, cnn=True, img=True, k=32)
    res = {}
    log('=== CV: GroupKFold(5) seed 0(=CNN fold) + seed 1,2, 캡, 확신 cummax ===')
    for name, cfg in cfgs.items():
        t0 = time.time()
        rows = run_cv(tr, cfg, probs, emb, seeds=[0, 1, 2])
        r = {k: (float(np.mean([x[k] for x in rows])), float(np.std([x[k] for x in rows]))) for k in ['raw', 'post']}
        r['seed0_post'] = rows[0]['post']
        r['per_class_post'] = np.mean([x['per_class'] for x in rows], 0).round(4).tolist()
        res[f'cv/{name}'] = r
        log(f'{name:16s} raw {r["raw"][0]:.4f}±{r["raw"][1]:.4f}  post {r["post"][0]:.4f}±{r["post"][1]:.4f} '
            f'(seed0 {r["seed0_post"]:.4f})  F1/cls {r["per_class_post"]}  ({time.time()-t0:.0f}s)')
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    log('\n=== Val (Train 전체 학습, 500행 캡) — CNN은 Val을 전혀 보지 않음 ===')
    vc = F.cap_rows(va).reset_index(drop=True)
    for name, cfg in cfgs.items():
        pca = fit_pca(tr, emb, cfg.get('k', 32)) if cfg.get('img') else None
        m = lgbm(42).fit(make_X(tr, cfg, probs, emb, pca), tr.label.astype(int))
        r = score6(vc, m.predict_proba(make_X(vc, cfg, probs, emb, pca)))
        res[f'val/{name}'] = r
        log(f'VAL {name:16s} raw {r["raw"]:.4f}  post {r["post"]:.4f}  F1/cls {r["per_class"]}')
    args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
