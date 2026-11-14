"""시간 구조 디코더 + 앙상블 실험 (oof_bank.py 결과 사용).

디코더 (모두 인과적: 현재 시점 이하의 확률만 사용)
- argmax
- conf_cummax τ=0.7 (현재 채택)
- hmm: 좌→우 HMM 순방향 필터링
    상태 1..6, 단계 k 체류시간 ~ 지수분포(평균 d_k, Train 학습 fold 실험의 라벨로 추정)
    Δt 동안 전이: 머무름 exp(-γΔt/d_k), 다음 단계로 이동(일부는 다음 단계 건너뛰기: 학습 fold에서 그 단계가 0초였던 비율)
    관측: 분류기 확률^α
    출력: 필터링 사후확률 argmax (→ 선택적으로 cummax)
- 체류시간을 bat_class 별로 추정하는 버전도 비교 (hmm_bat)

파라미터(α, γ)는 Train CV(3 seeds) 평균으로만 고르고, Val 은 마지막에 한 번 확인.

    python decode.py --data_dir <D> --banks v3 v3+cos v3+img_k32
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402
from exp_img import load, log  # noqa: E402

K = 6


def durations(frame):
    """실험별 단계 체류시간(초) → 단계별 평균, 건너뛴 비율."""
    rows = []
    for _, g in frame.groupby('experiment_id'):
        g = g.sort_values('relative_time_s')
        t = g.relative_time_s.to_numpy(float)
        y = g.label.astype(int).to_numpy()
        dt = np.diff(np.r_[t, t[-1] + 5])
        d = np.array([dt[y == k].sum() for k in range(1, K + 1)])
        rows.append(d)
    D = np.array(rows)
    mean = np.array([D[D[:, k] > 0, k].mean() if (D[:, k] > 0).any() else 60 for k in range(K)])
    skip = np.array([(D[:, k] == 0).mean() for k in range(K)])
    # 마지막 단계(6)는 실험 끝에서 잘려서 관측 체류가 짧음 → 흡수 상태
    return mean, skip


def make_params(frame, by_bat):
    if not by_bat:
        p = durations(frame)
        return {b: p for b in [1, 2, 3]}
    return {b: durations(frame[frame.bat_class == b]) for b in [1, 2, 3]}


def hmm_filter(P, t, mean, skip, alpha, gamma):
    n = len(P)
    E = np.clip(P, 1e-6, 1) ** alpha
    out = np.empty((n, K))
    a = np.zeros(K); a[0] = 1.0
    prev_t = t[0]
    for i in range(n):
        dt = max(t[i] - prev_t, 0.0); prev_t = t[i]
        if i > 0 and dt > 0:
            stay = np.exp(-gamma * dt / mean); stay[-1] = 1.0
            mv = a * (1 - stay)
            na = a * stay
            for k in range(K - 1):
                # 다음 단계가 0초였던 비율만큼은 그 다음으로 건너뜀
                s = skip[k + 1] if k + 2 < K else 0.0
                na[k + 1] += mv[k] * (1 - s)
                if k + 2 < K:
                    na[k + 2] += mv[k] * s
            a = na
        a = a * E[i]
        a = a / a.sum() if a.sum() > 0 else np.full(K, 1 / K)
        out[i] = a
    return out


def decode(frame, P, method, params=None, alpha=1.0, gamma=1.0, cm=False):
    """frame 행 순서대로 라벨 반환."""
    if method == 'argmax':
        return F.LABELS[P.argmax(1)]
    if method == 'conf':
        return F.postprocess(frame, P, 0.7)
    pred = np.empty(len(frame), int)
    t_all = frame.relative_time_s.to_numpy(float)
    bat = frame.bat_class.to_numpy()
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        mean, skip = params[int(bat[idx[0]])]
        Q = hmm_filter(P[idx], t_all[idx], mean, skip, alpha, gamma)
        p = Q.argmax(1) + 1
        pred[idx] = np.maximum.accumulate(p) if cm else p
    return pred


def f1(y, p):
    return f1_score(y, p, labels=F.LABELS, average='macro')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--banks', nargs='+', required=True)
    ap.add_argument('--out', type=Path, default=Path('res_decode.json'))
    args = ap.parse_args()
    tr, va = load(args.data_dir, 'train'), load(args.data_dir, 'validation')
    trs = tr.set_index('sample_id')
    banks = {b: np.load(f'bank_{b}.npz') for b in args.banks}
    seeds = sorted({int(k.split('_')[1][1:]) for k in banks[args.banks[0]].files if k.startswith('oof_s') and k.endswith('_P')})

    # 앙상블 후보: 각 단일 bank + 전체 평균 + 2개 조합 평균
    combos = [(b,) for b in args.banks] + [c for r in range(2, len(args.banks) + 1) for c in itertools.combinations(args.banks, r)]

    # seed 별 OOF 프레임 + fold 별 HMM 파라미터(그 fold 의 학습 실험 라벨로 추정)
    per_seed = {}
    for s in seeds:
        ids = banks[args.banks[0]][f'oof_s{s}_ids']
        frame = trs.loc[ids].reset_index()
        fold_of = {}
        params = {}
        for f, (a, b) in enumerate(GroupKFold(5, shuffle=True, random_state=s).split(tr, groups=tr.experiment_id)):
            for e in tr.experiment_id.iloc[b].unique():
                fold_of[e] = f
            params[f] = {False: make_params(tr.iloc[a], False), True: make_params(tr.iloc[a], True)}
        per_seed[s] = (frame, fold_of, params)

    def bank_P(combo, key):
        return np.mean([banks[b][key] for b in combo], 0)

    def cv_score(combo, method, by_bat=False, alpha=1.0, gamma=1.0, cm=False):
        sc = []
        for s in seeds:
            frame, fold_of, params = per_seed[s]
            P = bank_P(combo, f'oof_s{s}_P')
            for b in combo[1:]:
                assert (banks[b][f'oof_s{s}_ids'] == banks[combo[0]][f'oof_s{s}_ids']).all()
            y = frame.label.astype(int).to_numpy()
            if method in ('argmax', 'conf'):
                sc.append(f1(y, decode(frame, P, method)))
                continue
            pred = np.empty(len(frame), int)
            folds = frame.experiment_id.map(fold_of).to_numpy()
            for f in np.unique(folds):
                m = folds == f
                sub = frame[m].reset_index(drop=True)
                pred[m] = decode(sub, P[m], 'hmm', params[f][by_bat], alpha, gamma, cm)
            sc.append(f1(y, pred))
        return float(np.mean(sc)), float(np.std(sc))

    res = {}
    grid = list(itertools.product([0.5, 1.0, 1.5], [0.5, 1.0, 2.0, 4.0], [False, True], [False, True]))
    for combo in combos:
        name = '+'.join(combo)
        r = {'argmax': cv_score(combo, 'argmax'), 'conf': cv_score(combo, 'conf')}
        best = None
        for alpha, gamma, by_bat, cm in grid:
            sc = cv_score(combo, 'hmm', by_bat, alpha, gamma, cm)
            if best is None or sc[0] > best[1][0]:
                best = ((alpha, gamma, by_bat, cm), sc)
        r['hmm_best'] = dict(params=dict(alpha=best[0][0], gamma=best[0][1], by_bat=best[0][2], cummax=best[0][3]), cv=best[1])
        res[name] = r
        log(f'{name:28s} argmax {r["argmax"][0]:.4f} | conf {r["conf"][0]:.4f}±{r["conf"][1]:.4f} | '
            f'hmm {best[1][0]:.4f}±{best[1][1]:.4f} {r["hmm_best"]["params"]}')
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # CV 1위 (조합 × 디코더) 와 기준(v3 conf) 만 Val 확인
    cands = []
    for name, r in res.items():
        cands.append((r['conf'][0], name, 'conf', None))
        cands.append((r['hmm_best']['cv'][0], name, 'hmm', r['hmm_best']['params']))
    cands.sort(reverse=True)
    log('\nCV 순위 상위 5:')
    for c in cands[:5]:
        log(f'  {c[0]:.4f}  {c[1]}  {c[2]}  {c[3] or ""}')
    vframe = va.set_index('sample_id').loc[banks[args.banks[0]]['val_ids']].reset_index()
    yv = vframe.label.astype(int).to_numpy()
    full_params = {False: make_params(tr, False), True: make_params(tr, True)}
    checks = [cands[0]] + [c for c in cands if c[1] == 'v3' and c[2] == 'conf']
    res['_val'] = {}
    for sc, name, method, p in checks:
        P = np.mean([banks[b]['val_P'] for b in name_to_combo(name, args.banks)], 0)
        if method == 'conf':
            pred = decode(vframe, P, 'conf')
        else:
            pred = decode(vframe, P, 'hmm', full_params[p['by_bat']], p['alpha'], p['gamma'], p['cummax'])
        v = f1(yv, pred)
        pc = f1_score(yv, pred, labels=F.LABELS, average=None).round(4).tolist()
        res['_val'][f'{name}|{method}'] = dict(cv=sc, val=v, per_class=pc, params=p)
        log(f'VAL {name} {method}: {v:.4f}  F1/cls {pc}  (CV {sc:.4f})')
    args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


def name_to_combo(name, banks):
    """'v3+cos' 처럼 bank 이름에 '+' 가 있어 단순 split 불가 → 가장 긴 bank 이름부터 매칭."""
    out, rest = [], name
    while rest:
        for b in sorted(banks, key=len, reverse=True):
            if rest == b or rest.startswith(b + '+'):
                out.append(b)
                rest = rest[len(b) + 1:]
                break
        else:
            raise ValueError(name)
    return out


if __name__ == '__main__':
    main()
