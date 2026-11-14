"""이미지 임베딩(DINOv2, 고정) 효과 실험.

1단계: 2 vs 3 이진 분류 (2·3단계 행만) — 시계열 피처 vs 이미지 피처 vs 둘 다
2단계: v3 시계열 피처(735) + 이미지 시계열 피처 → 6클래스 LightGBM

평가 규칙 (README 5-3)
- 선택은 Train CV로만: 78개 실험 GroupKFold(5) × 시드 3, 검증 fold는 500행 캡 + 확신 cummax τ=0.7
- PCA는 각 fold의 학습 실험 프레임으로만 fit (Val/Test 프레임은 transform만)
- Val(19개 실험)은 CV로 고른 최종 후보와 v3 기준선만 마지막에 확인

    python exp_img.py --data_dir <battery_data_light> --emb emb_dinov2s.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402

SEEDS = [0, 1, 2]
IMG_WINDOWS = [30, 60, 120]


def log(*a):
    print(*a, flush=True)


def load(d, split):
    return pd.read_csv(Path(d) / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})


# ---------------------------------------------------------------------------
# 이미지 피처 (인과적: 같은 실험의 현재 시점 이하 프레임만 사용)
# ---------------------------------------------------------------------------
def img_features(frame, emb, pca, k, windows=IMG_WINDOWS, use_pca=True, use_cos=True):
    E = emb.loc[frame.sample_id.values].to_numpy(np.float32)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    Z = pca.transform(E)[:, :k].astype(np.float32) if use_pca else None
    t_all = frame.relative_time_s.to_numpy(float)
    cols = {}
    n = len(frame)

    def put(name, idx, vals):
        if name not in cols:
            cols[name] = np.full(n, np.nan, np.float32)
        cols[name][idx] = vals

    for _, idx in frame.reset_index(drop=True).groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        t = t_all[idx]
        en = En[idx]
        if use_cos:
            cf = 1 - en @ en[0]                          # 첫 프레임(t=0)과의 코사인 거리
            put('img_cos_first', idx, cf)
            put('img_cos_first_cummax', idx, np.maximum.accumulate(cf))
            put('img_cos_prev', idx, np.r_[np.nan, 1 - (en[1:] * en[:-1]).sum(1)])
        if use_pca:
            z = Z[idx]
            for j in range(k):
                put(f'img_z{j}', idx, z[:, j])
            cz = np.vstack([np.zeros((1, k)), np.cumsum(z, 0)])
        for w in windows:
            lo = np.searchsorted(t, t - w, side='right')           # 윈도우 (t-w, t]
            lag = np.searchsorted(t, t - w, side='right') - 1      # t-w 이하 마지막 관측
            ok = lag >= 0
            lagc = np.clip(lag, 0, None)
            if use_cos:
                c = 1 - (en * en[lagc]).sum(1)
                put(f'img_cos_lag{w}', idx, np.where(ok, c, np.nan))
                ccf = np.r_[0, np.cumsum(cf)]
                put(f'img_cos_first_mean{w}', idx, (ccf[np.arange(len(t)) + 1] - ccf[lo]) / (np.arange(len(t)) + 1 - lo))
            if use_pca:
                cnt = (np.arange(len(t)) + 1 - lo)[:, None]
                m = (cz[np.arange(len(t)) + 1] - cz[lo]) / cnt
                dz = np.where(ok[:, None], z - z[lagc], np.nan)
                for j in range(k):
                    put(f'img_z{j}_mean{w}', idx, m[:, j])
                    put(f'img_z{j}_delta{w}', idx, dz[:, j])
                put(f'img_zdrift{w}', idx, np.where(ok, np.linalg.norm(z - z[lagc], axis=1), np.nan))
    return pd.DataFrame(cols, index=frame.sample_id.values)


# ---------------------------------------------------------------------------
def lgbm(seed, binary=False):
    """GBM 분류기. 환경변수 GBM_BACKEND = lgbm_cpu(기본) | lgbm_cuda | xgb_cuda"""
    import os
    backend = os.environ.get('GBM_BACKEND', 'lgbm_cpu')
    if backend == 'xgb_cuda':
        return XGBBalanced(seed)
    import lightgbm as lgb
    extra = dict(device='cuda') if backend == 'lgbm_cuda' else {}
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
        class_weight='balanced', random_state=seed, n_jobs=-1, verbose=-1, **extra)


class XGBBalanced:
    """XGBoost(GPU hist) 를 LightGBM 설정과 최대한 맞춘 래퍼. class_weight='balanced' 를 sample_weight 로 구현."""

    def __init__(self, seed):
        import xgboost as xgb
        self.m = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_leaves=31, max_depth=0, grow_policy='lossguide',
            min_child_weight=50, subsample=0.8, colsample_bytree=0.3, reg_lambda=1.0,
            tree_method='hist', device='cuda', random_state=seed, n_jobs=8, verbosity=0)

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        cnt = np.array([(y == c).sum() for c in self.classes_], float)
        w = (len(y) / (len(self.classes_) * cnt))[np.searchsorted(self.classes_, y)]
        self.m.fit(X.to_numpy(np.float32) if hasattr(X, 'to_numpy') else X, np.searchsorted(self.classes_, y), sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X.to_numpy(np.float32) if hasattr(X, 'to_numpy') else X)


def folds(tr, seed):
    return GroupKFold(5, shuffle=True, random_state=seed).split(tr, groups=tr.experiment_id)


def make_X(frame, emb, pca, cfg):
    parts = []
    if cfg['ts']:
        parts.append(F.build_features(frame))
    if cfg.get('img'):
        parts.append(img_features(frame, emb, pca, cfg.get('k', 32),
                                  use_pca=cfg.get('pca', True), use_cos=cfg.get('cos', True)))
    if cfg.get('raw_z'):   # 현재 프레임 임베딩만 (윈도우 없음)
        Z = pca.transform(emb.loc[frame.sample_id.values].to_numpy(np.float32))[:, :cfg['k']]
        parts.append(pd.DataFrame(Z, index=frame.sample_id.values, columns=[f'z{j}' for j in range(cfg['k'])]))
    X = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
    return X.loc[frame.sample_id.values]


def fit_pca(frame, emb, k):
    E = emb.loc[frame.sample_id.values].to_numpy(np.float32)
    return PCA(k, random_state=0).fit(E)


def run_cv(tr, emb, cfg, binary=False):
    """반환: seed별 OOF 확률 리스트(행 순서는 캡된 fold 행들을 이어붙인 순서) + 해당 프레임."""
    out = []
    for s in SEEDS:
        frames, probs = [], []
        for a, b in folds(tr, s):
            ftr, fev = tr.iloc[a], F.cap_rows(tr.iloc[b])
            pca = fit_pca(ftr, emb, cfg.get('k', 32)) if (cfg.get('img') or cfg.get('raw_z')) else None
            Xtr, Xev = make_X(ftr, emb, pca, cfg), make_X(fev, emb, pca, cfg)
            ytr = ftr.label.astype(int).to_numpy()
            if binary:   # 2·3단계 행만 학습/평가
                mtr = np.isin(ytr, [2, 3]); mev = fev.label.isin([2, 3]).to_numpy()
                m = lgbm(s).fit(Xtr[mtr], ytr[mtr])
                frames.append(fev[mev]); probs.append(m.predict_proba(Xev[mev])[:, 1])
            else:
                m = lgbm(s).fit(Xtr, ytr)
                frames.append(fev); probs.append(m.predict_proba(Xev))
        out.append((pd.concat(frames), np.concatenate(probs)))
    return out


def score6(frame, P):
    y = frame.label.astype(int).to_numpy()
    raw = F.LABELS[P.argmax(1)]
    post = F.postprocess(frame.reset_index(drop=True), P, 0.7)
    return dict(raw=f1_score(y, raw, labels=F.LABELS, average='macro'),
                post=f1_score(y, post, labels=F.LABELS, average='macro'),
                per_class=f1_score(y, post, labels=F.LABELS, average=None).round(4).tolist())


def score2(frame, p):
    y = (frame.label.astype(int).to_numpy() == 3).astype(int)
    return dict(f1=f1_score(y, (p >= 0.5).astype(int), average='macro'), auc=roc_auc_score(y, p))


def agg(rows, keys):
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--emb', type=Path, required=True)
    ap.add_argument('--emb_kind', default='both', choices=['cls', 'patch_mean', 'both'])
    ap.add_argument('--stage', default='12', help='1, 2, 12')
    ap.add_argument('--out', type=Path, default=Path('exp_img_results.json'))
    args = ap.parse_args()

    z = np.load(args.emb)
    E = np.concatenate([z['cls'], z['patch_mean']], 1) if args.emb_kind == 'both' else z[args.emb_kind]
    emb = pd.DataFrame(E.astype(np.float32), index=z['sample_id'].astype(str))
    tr, va = load(args.data_dir, 'train'), load(args.data_dir, 'validation')
    log(f'train {len(tr)} rows / {tr.experiment_id.nunique()} exps, val {len(va)}, emb {E.shape}')
    res = {}

    if '1' in args.stage:
        log('\n=== 1단계: 2 vs 3 이진 분류 (CV, 2·3단계 행만, 캡) ===')
        cfgs = {
            'ts_v3': dict(ts=True),
            'img_raw_z64': dict(ts=False, raw_z=True, k=64),
            'img_feats_k32': dict(ts=False, img=True, k=32),
            'ts_v3+img_k32': dict(ts=True, img=True, k=32),
        }
        for name, cfg in cfgs.items():
            t0 = time.time()
            rows = [score2(f, p) for f, p in run_cv(tr, emb, cfg, binary=True)]
            res[f'bin/{name}'] = agg(rows, ['f1', 'auc'])
            r = res[f'bin/{name}']
            log(f'{name:18s} F1 {r["f1"][0]:.4f}±{r["f1"][1]:.4f}  AUC {r["auc"][0]:.4f}±{r["auc"][1]:.4f}  ({time.time()-t0:.0f}s)')
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    if '2' in args.stage:
        log('\n=== 2단계: 6클래스, v3 + 이미지 피처 (CV GroupKFold(5)×3 seeds, 캡, 확신 cummax) ===')
        cfgs = {
            'v3': dict(ts=True),
            'v3+cos': dict(ts=True, img=True, pca=False, cos=True),
            'v3+img_k32': dict(ts=True, img=True, k=32),
            'v3+img_k64': dict(ts=True, img=True, k=64),
        }
        for name, cfg in cfgs.items():
            t0 = time.time()
            rows = [score6(f, P) for f, P in run_cv(tr, emb, cfg)]
            r = agg(rows, ['raw', 'post'])
            r['per_class_post'] = np.mean([x['per_class'] for x in rows], 0).round(4).tolist()
            res[f'cv6/{name}'] = r
            log(f'{name:12s} raw {r["raw"][0]:.4f}±{r["raw"][1]:.4f}  post {r["post"][0]:.4f}±{r["post"][1]:.4f}  '
                f'F1/cls {r["per_class_post"]}  ({time.time()-t0:.0f}s)')
            args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

        # CV로 고른 후보 + 기준선만 Val 확인
        best = max([k for k in cfgs if k != 'v3'], key=lambda k: res[f'cv6/{k}']['post'][0])
        log(f'\nCV 최고 이미지 설정: {best} → Val 확인 (Train 전체 학습, Val 500행 캡)')
        vc = F.cap_rows(va).reset_index(drop=True)
        for name in ['v3', best]:
            cfg = cfgs[name]
            pca = fit_pca(tr, emb, cfg.get('k', 32)) if cfg.get('img') else None
            m = lgbm(42).fit(make_X(tr, emb, pca, cfg), tr.label.astype(int))
            r = score6(vc, m.predict_proba(make_X(vc, emb, pca, cfg)))
            res[f'val/{name}'] = r
            log(f'VAL {name:12s} raw {r["raw"]:.4f}  post {r["post"]:.4f}  F1/cls {r["per_class"]}')
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
