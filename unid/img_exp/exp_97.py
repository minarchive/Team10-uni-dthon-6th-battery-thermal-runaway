"""다른 서버 결과(385피처, Train+Val 97실험 5-fold, 500행 캡, prefix Viterbi 0.9146 / raw 0.9130)와
같은 평가 방식으로 기존 피처의 기준선을 잰다.

피처 세트
  v3      : val0.9195 build_features 735개
  v3-gas  : v3에서 가스 10종 + gas_sum 관련 열(신호당 35개 × 11 = 385개) 제거 → 350개
평가
  Train+Val 97개 실험, GroupKFold(5, shuffle, seed) 실험 단위. 시드 여러 개.
  학습: 학습 fold 실험의 전체 행. 평가: 평가 fold 실험을 500행 캡한 행.
    --cap_first 0 : 전체 행으로 피처 계산 후 캡 (기존 파이프라인 방식)
    --cap_first 1 : 캡한 행만으로 피처 계산 (Test와 완전히 같은 조건)
  디코딩: raw argmax / 확신 cummax τ=0.7 / prefix Viterbi (각 시점까지의 단조 비감소 최적 경로의 마지막 상태, 인과적)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402
from exp_img import lgbm  # GBM_BACKEND=xgb_cuda (이 venv의 LightGBM은 libnccl 없이 로드 불가)  # noqa: E402

GAS_SIG = F.GAS + ['gas_sum']


def prefix_viterbi(P):
    """인과적: 시점 t의 출력 = 0..t 구간에서 비감소 라벨 경로 중 log p 합이 최대인 경로의 마지막 라벨."""
    L = np.log(np.clip(P, 1e-6, 1))
    D = L[0].copy()
    out = np.empty(len(P), int)
    out[0] = D.argmax()
    for t in range(1, len(P)):
        D = np.maximum.accumulate(D) + L[t]          # 이전 라벨 ≤ 현재 라벨
        out[t] = D.argmax()
    return F.LABELS[out]


def decode(frame, P, mode):
    if mode == 'raw':
        return F.LABELS[P.argmax(1)]
    if mode == 'cummax':
        return F.postprocess(frame, P, 0.7)
    out = np.empty(len(frame), int)
    t = frame.relative_time_s.to_numpy(float)
    for _, idx in frame.reset_index(drop=True).groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        out[idx] = prefix_viterbi(P[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, default=Path('/home/nxcolelxu/unid/data/battery_data_light'))
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--cap_first', type=int, default=1)
    ap.add_argument('--n_jobs', type=int, default=8)
    ap.add_argument('--out', default='res_97.json')
    a = ap.parse_args()
    rd = lambda s: pd.read_csv(a.data_dir / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    df = pd.concat([rd('train'), rd('validation')], ignore_index=True)
    df['label'] = df.label.astype(int)
    cap = F.cap_rows(df).copy()
    t0 = time.time()
    X = F.build_features(df)
    Xc = F.build_features(cap) if a.cap_first else X.loc[cap.sample_id.astype(str).values]
    print(f'{df.experiment_id.nunique()} exps, {len(df)} rows, cap {len(cap)} rows, features {X.shape} ({time.time()-t0:.0f}s)', flush=True)
    gas_cols = [c for c in X.columns if any(c == g or c.startswith(g + '_') for g in GAS_SIG)]
    sets = {'v3': list(X.columns), 'v3-gas': [c for c in X.columns if c not in set(gas_cols)]}
    print({k: len(v) for k, v in sets.items()}, 'gas cols', len(gas_cols), flush=True)

    y, yc = df.label.to_numpy(), cap.label.to_numpy()
    exps = df.experiment_id.to_numpy()
    uexp = np.unique(exps)
    res = {}
    for name, cols in sets.items():
        for seed in [int(s) for s in a.seeds.split(',')]:
            P = np.zeros((len(cap), 6))
            for _, te in GroupKFold(5, shuffle=True, random_state=seed).split(uexp, groups=uexp):
                te_e = set(uexp[te])
                tr_m = ~np.isin(exps, list(te_e))
                ev_m = cap.experiment_id.isin(te_e).to_numpy()
                m = lgbm(seed)
                m.fit(X.loc[tr_m, cols], y[tr_m])
                P[ev_m] = m.predict_proba(Xc.loc[ev_m, cols])
            r = {}
            for mode in ('raw', 'cummax', 'viterbi'):
                lab = decode(cap, P, mode)
                r[mode] = float(f1_score(yc, lab, labels=F.LABELS, average='macro'))
                r[mode + '_pc'] = f1_score(yc, lab, labels=F.LABELS, average=None).round(3).tolist()
            res[f'{name}/s{seed}'] = r
            print(f'{name:7s} seed {seed}: raw {r["raw"]:.4f}  cummax {r["cummax"]:.4f}  viterbi {r["viterbi"]:.4f}  '
                  f'viterbi per-class {r["viterbi_pc"]}  ({time.time()-t0:.0f}s)', flush=True)
        for mode in ('raw', 'cummax', 'viterbi'):
            v = [res[f'{name}/s{s}'][mode] for s in a.seeds.split(',')]
            print(f'  {name} {mode}: {np.mean(v):.4f} ± {np.std(v):.4f}', flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
