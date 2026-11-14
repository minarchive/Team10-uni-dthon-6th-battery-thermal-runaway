"""유사 실험 검색(retrieval) prior 피처 실험.

아이디어: 샘플이 들어오면 학습 데이터에서 '비슷한 실험'을 찾고,
그 실험들의 단계 진행(특히 2단계 지속시간, 3단계 시작 온도)을 prior로 피처에 넣는다.

변형
- base : v3 피처 (735개)
- cfg  : + 같은 메타 조합(bat_class, bat_cc, soc_status, tr_method) 학습 실험들의 prior
- knn5 : + 메타 + 45°C 도달 직전까지의 궤적으로 찾은 최근접 5개 실험의 prior
- knn1 : + 최근접 1개 실험의 prior ("이 샘플은 학습 데이터의 어느 실험인가")
- cfg+knn5

누수 방지
- prior는 참조(학습) 실험의 라벨로만 계산한다.
- 학습 fold 행의 prior는 자기 실험을 참조에서 뺀 상태(leave-one-experiment-out)로 계산한다.
- 대상 실험 쪽 디스크립터는 45°C 최초 도달 시점까지의 관측만 쓰고, 도달 전 행은 NaN (인과적).

평가: Train GroupKFold(5) × 시드 3개, 검증 fold 500행 캡, 확신 cummax τ=0.7, Macro F1.
Val은 참고용으로만 출력 (선택에 사용하지 않음).
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import ts_model as M
from exp_post import _per_exp, conf_cummax, load

L = np.array(M.LABELS)
META = M.META
DESC_CONT = ['log_t45', 'pre_rate60', 'pre_rate120', 't1_start', 'v0', 'log_cc']


def f1(y, p):
    return f1_score(y, p, labels=M.LABELS, average='macro', zero_division=0)


# ------------------------------------------------------------ 실험 단위 정보
def descriptors(frame):
    """실험별 디스크립터. 45°C 최초 도달 시점(t45)까지의 관측만 사용."""
    rows = {}
    for e, g in frame.sort_values('relative_time_s').groupby('experiment_id'):
        t = g.relative_time_s.to_numpy(float)
        t1 = g.meas_t1.to_numpy(float)
        cm = np.fmax.accumulate(np.nan_to_num(t1, nan=-1e9))
        hit = np.where(cm >= 45)[0]
        r = {m: float(g[m].iloc[0]) for m in META}
        r['log_cc'] = np.log1p(r['bat_cc'])
        r['t1_start'] = t1[0]
        r['v0'] = float(g.meas_v.iloc[0])
        if len(hit):
            i = hit[0]; t45 = t[i]
            r['t45'] = t45; r['log_t45'] = np.log1p(t45)
            for w in (60, 120):
                j = np.searchsorted(t, t45 - w)
                r[f'pre_rate{w}'] = (t1[i] - t1[j]) / max(t45 - t[j], 5)
        else:
            r['t45'] = np.nan
        rows[e] = r
    return pd.DataFrame.from_dict(rows, orient='index')


def outcomes(frame):
    """참조 실험의 라벨 기반 결과값 (학습 실험에만 계산)."""
    rows = {}
    for e, g in frame.sort_values('relative_time_s').groupby('experiment_id'):
        t, l = g.relative_time_s.to_numpy(float), g.label.to_numpy()
        cm = np.fmax.accumulate(np.nan_to_num(g.meas_t1.to_numpy(float), nan=-1e9))
        t12, i3 = t[l >= 2].min(), np.argmax(l >= 3)
        dur2 = t[i3] - t12
        rows[e] = dict(dur2=dur2, t1at3=cm[i3], rate2=(cm[i3] - 45) / max(dur2, 5))
    return pd.DataFrame.from_dict(rows, orient='index')


def neighbors(D, tgt, refs, k):
    """tgt 실험에 대해 refs 중 최근접 k개. 메타 불일치는 큰 거리 페널티."""
    R = D.loc[refs]
    mu, sd = R[DESC_CONT].mean(), R[DESC_CONT].std().replace(0, 1)
    z = ((R[DESC_CONT] - mu) / sd).fillna(0).to_numpy()
    zt = ((D.loc[[tgt], DESC_CONT] - mu) / sd).fillna(0).to_numpy()
    dist = np.sqrt(((z - zt) ** 2).sum(1)) + 3.0 * (R[META].to_numpy() != D.loc[tgt, META].to_numpy().astype(float)).sum(1)
    o = np.argsort(dist, kind='mergesort')[:k]
    return [refs[i] for i in o]


def prior_table(D, O, targets, refs_fn, mode):
    """대상 실험별 prior 값. refs_fn(e) -> 참조 실험 목록."""
    rows = {}
    for e in targets:
        refs = refs_fn(e)
        if mode == 'cfg':
            same = [r for r in refs if (D.loc[r, META].to_numpy() == D.loc[e, META].to_numpy()).all()]
            if not same:
                same = [r for r in refs if D.loc[r, 'bat_class'] == D.loc[e, 'bat_class']]
            nb = same
        else:
            nb = neighbors(D, e, refs, int(mode[3:]))
        o = O.loc[nb]
        rows[e] = {f'{mode}_dur2_med': o.dur2.median(), f'{mode}_dur2_q25': o.dur2.quantile(.25),
                   f'{mode}_dur2_q75': o.dur2.quantile(.75), f'{mode}_t1at3_med': o.t1at3.median(),
                   f'{mode}_rate2_med': o.rate2.median(), f'{mode}_n': len(nb)}
    return pd.DataFrame.from_dict(rows, orient='index')


def add_prior(X, frame, P, D, mode):
    """행 단위로 prior + 현재 진행 상태와의 상호작용 피처를 붙인다 (45°C 도달 전 행은 NaN)."""
    pr = P.reindex(frame.experiment_id.astype(str).values)
    pr.index = X.index
    el = X['meas_t1_since45']                       # 45°C 도달 이후 경과 시간 (도달 전 NaN)
    cm1 = X['t1_rise_since45'] + 45
    reached = el.notna()
    out = pr.astype(float)
    out.loc[~reached.values] = np.nan
    med = out[f'{mode}_dur2_med']
    out[f'{mode}_prog'] = el / med.clip(lower=5)
    out[f'{mode}_left'] = med - el
    out[f'{mode}_prog_q25'] = el / out[f'{mode}_dur2_q25'].clip(lower=5)
    out[f'{mode}_prog_q75'] = el / out[f'{mode}_dur2_q75'].clip(lower=5)
    out[f'{mode}_t1gap'] = cm1 - out[f'{mode}_t1at3_med']
    out[f'{mode}_t1exp'] = cm1 - (45 + out[f'{mode}_rate2_med'] * el)
    return pd.concat([X, out], axis=1)


# ------------------------------------------------------------ 실험 루프
DEVICE = 'cpu'


def make_model():
    m = M.build_model('lgbm', 'balanced')
    if DEVICE != 'cpu':
        m.set_params(device_type=DEVICE)
    return m


def run(tr, va, variants, seeds):
    t0 = time.time()
    trc = M.cap_rows(tr)
    X_full = M.build_features(tr); X_full.index = tr.index
    X_cap = M.build_features(trc); X_cap.index = trc.index
    D = descriptors(pd.concat([tr, va]))
    O = outcomes(tr)
    exps = np.array(sorted(tr.experiment_id.astype(str).unique()))
    print(f'피처 준비 {time.time() - t0:.0f}s, base {X_full.shape[1]}개', flush=True)

    res = {v: [] for v in variants}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        fold_of = dict(zip(exps, rng.permutation(len(exps)) % 5))
        f_tr = tr.experiment_id.astype(str).map(fold_of).to_numpy()
        f_cap = trc.experiment_id.astype(str).map(fold_of).to_numpy()
        oof = {v: np.zeros((len(trc), 6)) for v in variants}
        for k in range(5):
            train_exps = [e for e in exps if fold_of[e] != k]
            test_exps = [e for e in exps if fold_of[e] == k]
            a, b = np.where(f_tr != k)[0], np.where(f_cap == k)[0]
            ftr, fte = tr.iloc[a], trc.iloc[b]
            for v in variants:
                Xa, Xb = X_full.iloc[a], X_cap.iloc[b]
                for mode in [m for m in v.split('+') if m != 'base']:
                    Ptr = prior_table(D, O, train_exps, lambda e: [r for r in train_exps if r != e], mode)
                    Pte = prior_table(D, O, test_exps, lambda e: train_exps, mode)
                    Xa, Xb = add_prior(Xa, ftr, Ptr, D, mode), add_prior(Xb, fte, Pte, D, mode)
                m = make_model().fit(Xa, ftr.label)
                oof[v][b] = m.predict_proba(Xb)
        for v in variants:
            pred = _per_exp(trc, oof[v], lambda P, t: conf_cummax(P, 0.7))
            raw = L[oof[v].argmax(1)]
            res[v].append(dict(seed=seed, cv=f1(trc.label, pred), cv_raw=f1(trc.label, raw),
                               c2=f1_score(trc.label, pred, labels=[2], average=None)[0],
                               c3=f1_score(trc.label, pred, labels=[3], average=None)[0]))
            print(f'  seed {seed} {v:10s} CV {res[v][-1]["cv"]:.4f} (raw {res[v][-1]["cv_raw"]:.4f}) '
                  f'L2 {res[v][-1]["c2"]:.3f} L3 {res[v][-1]["c3"]:.3f}  [{time.time() - t0:.0f}s]', flush=True)

    # Val 참고 (Train 전체로 학습, 참조 = Train 78개)
    vc = M.cap_rows(va)
    Xv = M.build_features(vc); Xv.index = vc.index
    val = {}
    for v in variants:
        Xa, Xb = X_full, Xv
        for mode in [m for m in v.split('+') if m != 'base']:
            Ptr = prior_table(D, O, list(exps), lambda e: [r for r in exps if r != e], mode)
            Pva = prior_table(D, O, sorted(vc.experiment_id.astype(str).unique()), lambda e: list(exps), mode)
            Xa, Xb = add_prior(Xa, tr, Ptr, D, mode), add_prior(Xb, vc, Pva, D, mode)
        m = make_model().fit(Xa, tr.label)
        Pv = m.predict_proba(Xb)
        pred = _per_exp(vc, Pv, lambda P, t: conf_cummax(P, 0.7))
        val[v] = dict(val=f1(vc.label, pred), val_c2=f1_score(vc.label, pred, labels=[2], average=None)[0])

    rows = []
    for v in variants:
        r = pd.DataFrame(res[v])
        rows.append(dict(variant=v, cv=r.cv.mean(), cv_sd=r.cv.std(), cv_raw=r.cv_raw.mean(),
                         L2=r.c2.mean(), L3=r.c3.mean(), **val[v]))
    df = pd.DataFrame(rows)
    df['d_cv'] = df.cv - df.cv.iloc[0]
    # 시드별 짝 비교 (같은 fold 분할)
    for v in variants[1:]:
        d = np.array([x['cv'] for x in res[v]]) - np.array([x['cv'] for x in res[variants[0]]])
        print(f'{v}: 시드별 ΔCV {np.round(d, 4).tolist()}')
    return df


def check_prior_quality(tr, va):
    """prior가 2단계 지속시간을 얼마나 잘 맞히는지 (실험 단위, LOO on Train → Val)."""
    D = descriptors(pd.concat([tr, va])); O = outcomes(pd.concat([tr, va]))
    exps = sorted(tr.experiment_id.astype(str).unique())
    vexps = sorted(va.experiment_id.astype(str).unique())
    print('\n[prior 품질] 2단계 지속시간 예측 MAE(초)')
    for name, tg, refs_fn in [('Train LOO', exps, lambda e: [r for r in exps if r != e]),
                              ('Val (참조=Train)', vexps, lambda e: exps)]:
        line = [f'{name:16s} 전체중앙값 {np.mean([abs(O.loc[refs_fn(e)].dur2.median() - O.loc[e].dur2) for e in tg]):.0f}']
        for mode in ['cfg', 'knn1', 'knn3', 'knn5', 'knn10']:
            P = prior_table(D, O, tg, refs_fn, mode)
            line.append(f'{mode} {np.mean(np.abs(P[f"{mode}_dur2_med"] - O.loc[tg].dur2)):.0f}')
        print('  ' + ' | '.join(line))
    print('  상관(spearman) dur2 vs 디스크립터:',
          D.loc[exps + vexps, DESC_CONT].corrwith(O.loc[exps + vexps].dur2, method='spearman').round(2).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='/home/ubuntu/data/battery_data_light')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--device', default='cpu', help="LightGBM device_type: cpu / cuda")
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device
    tr, va = load(args.data_dir, 'train'), load(args.data_dir, 'validation')
    check_prior_quality(tr, va)
    variants = ['base', 'base+cfg', 'base+knn5', 'base+knn1', 'base+cfg+knn5']
    df = run(tr, va, variants, [int(s) for s in args.seeds.split(',')])
    pd.set_option('display.width', 200)
    print('\n', df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
