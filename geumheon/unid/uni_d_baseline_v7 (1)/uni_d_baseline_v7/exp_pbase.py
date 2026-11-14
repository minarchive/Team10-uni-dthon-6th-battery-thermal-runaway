"""압력 기준선(시작값) 실험.

변형(V)
- abs : 현재 모델 (meas_p 절대값 기반 피처 포함)
- rel : meas_p 를 (meas_p - p0) 로 바꿔서 같은 피처를 생성 (절대 압력 피처 없음)
- both: abs + rel 압력 피처를 모두 사용 (rel 은 prel_ 접두사)
- nop : 압력 관련 피처를 전부 제거 (대조군)
p0 = 실험 시작 후 30초 이내 관측의 중앙값(그 시점까지의 관측만 사용, 인과적).

평가 (선택은 Train 만 사용. Val 은 마지막 확인용)
- cv  : Train 78개 실험, bat_cc 계층화 GroupKFold(5) x 3 seeds, 검증 fold 500행 캡
- loco: leave-one-bat_cc-out (6 fold) — 처음 보는 셀 종류/압력 기준선에 대한 견고성
- val : Train 전체로 학습 → Validation(500행 캡)
후처리: raw / 확신 cummax τ=0.7 (이미 Train CV 로 선택된 방법)
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import ts_model as M
from exp_post import conf_cummax, _per_exp, load

L = M.LABELS
P_WIN_S = 30
SEEDS = [0, 1, 2]


def add_rel_pressure(frame):
    """meas_p 를 meas_p - p0 로 교체한 복사본. p0 는 t<=30s 관측의 (그 시점까지) 중앙값."""
    f = frame.copy()
    t = f['relative_time_s'].to_numpy(float)
    p = f['meas_p'].to_numpy(float)
    out = np.full(len(f), np.nan)
    for _, idx in f.groupby('experiment_id', sort=False).indices.items():
        o = idx[np.argsort(t[idx], kind='stable')]
        ps, ts = p[o], t[o]
        n_early = max(int((ts <= P_WIN_S).sum()), 1)
        p0 = np.empty(len(ps))
        for i in range(n_early):
            p0[i] = np.nanmedian(ps[:i + 1])
        p0[n_early:] = p0[n_early - 1]
        out[o] = ps - p0
    f['meas_p'] = out
    return f


def is_p(col):
    return col == 'meas_p' or col.startswith('meas_p_')


def feats(frame, variant):
    X = M.build_features(frame)
    if variant == 'abs':
        return X
    if variant == 'nop':
        return X[[c for c in X.columns if not is_p(c)]]
    R = M.build_features(add_rel_pressure(frame))
    pc = [c for c in R.columns if is_p(c)]
    if variant == 'rel':
        X = X[[c for c in X.columns if not is_p(c)]]
        return pd.concat([X, R[pc]], axis=1)
    if variant == 'both':
        return pd.concat([X, R[pc].add_prefix('prel_')], axis=1)
    raise ValueError(variant)


def strat_group_folds(exp_df, seed, k=5):
    """bat_cc 별로 실험을 섞어 fold 에 라운드로빈 배정."""
    rng = np.random.RandomState(seed)
    fold = {}
    for _, g in exp_df.groupby('bat_cc'):
        ids = list(g.experiment_id)
        rng.shuffle(ids)
        off = rng.randint(k)
        for r, e in enumerate(ids):
            fold[e] = (r + off) % k
    return fold


def macro(y, p):
    present = [c for c in L if (np.asarray(y) == c).any()]
    return f1_score(y, p, labels=present, average='macro', zero_division=0)


def post(frame, P, how):
    if how == 'raw':
        return np.array(L)[P.argmax(1)]
    return _per_exp(frame, P, lambda Pe, t: conf_cummax(Pe, 0.7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='/home/ubuntu/data/battery_data_light')
    ap.add_argument('--variants', default='abs,rel,both,nop')
    ap.add_argument('--out', default='exp_pbase')
    args = ap.parse_args()
    tr = load(args.data_dir, 'train'); va = load(args.data_dir, 'validation')
    trc, vac = M.cap_rows(tr), M.cap_rows(va)           # 평가용 500행 캡 프레임
    exps = tr.drop_duplicates('experiment_id')[['experiment_id', 'bat_cc', 'bat_class']]
    summary, store = {}, {}

    for v in args.variants.split(','):
        t0 = time.time()
        Xf, Xc, Xv = feats(tr, v), feats(trc, v), feats(vac, v)   # 학습=캡 없음, 평가=캡
        print(f'[{v}] features {Xf.shape[1]}  ({time.time() - t0:.0f}s)', flush=True)
        eid_f, eid_c = tr.experiment_id.to_numpy(), trc.experiment_id.to_numpy()
        yf = tr.label.to_numpy()

        # --- CV (bat_cc 계층화, 3 seeds)
        cv_probs = []
        for s in SEEDS:
            fold = strat_group_folds(exps, s)
            Pc = np.zeros((len(trc), 6))
            fc = np.array([fold[e] for e in eid_c]); ff = np.array([fold[e] for e in eid_f])
            for k in range(5):
                m = M.build_model('lgbm', 'balanced').fit(Xf[ff != k], yf[ff != k])
                Pc[fc == k] = m.predict_proba(Xc[fc == k])
            cv_probs.append(Pc)
        # --- LOCO
        Pl = np.zeros((len(trc), 6))
        cc_f, cc_c = tr.bat_cc.to_numpy(), trc.bat_cc.to_numpy()
        for cc in sorted(np.unique(cc_f)):
            m = M.build_model('lgbm', 'balanced').fit(Xf[cc_f != cc], yf[cc_f != cc])
            Pl[cc_c == cc] = m.predict_proba(Xc[cc_c == cc])
        # --- Val
        m = M.build_model('lgbm', 'balanced').fit(Xf, yf)
        Pv = m.predict_proba(Xv)

        r = {}
        for how in ['raw', 'conf0.7']:
            cvs = [macro(trc.label, post(trc, P, how)) for P in cv_probs]
            pl = post(trc, Pl, how)
            r[how] = dict(
                cv_mean=float(np.mean(cvs)), cv_std=float(np.std(cvs)),
                loco_pooled=float(macro(trc.label, pl)),
                loco_by_cc={str(cc): float(macro(trc.label[cc_c == cc], pl[cc_c == cc])) for cc in sorted(np.unique(cc_c))},
                val=float(macro(vac.label, post(vac, Pv, how))))
        r['n_feat'] = int(Xf.shape[1]); r['sec'] = round(time.time() - t0)
        summary[v] = r
        store[f'{v}_cv'] = np.stack(cv_probs); store[f'{v}_loco'] = Pl; store[f'{v}_val'] = Pv
        print(json.dumps({v: r}, indent=1), flush=True)
        np.savez(f'{args.out}_probs.npz', **store)
        Path(f'{args.out}.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
