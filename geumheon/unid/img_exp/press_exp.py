"""압력 시작점(p0) 문제 실험: Train+Val 합친 97개 실험 5-fold CV (평가 fold 500행 캡 = Test 조건).

SOTA 0.8812 피처·LightGBM 설정·인과적 단조 Viterbi 를 그대로 쓰고, 압력 처리만 바꾼다.
이미지는 반드시 섞는다: 기존 ConvNeXt-tiny 확률(ft_convnext_tiny.npz), 기본 α=0.1 (참고로 0.06 도 출력).

변형 (센서 LightGBM):
  ref_AC     SOTA: A(166) + 과충전 행만 C(압력 절대값·v_peak_ratio 제외)
  C_all      모든 행 C
  C_rel      C + 기준점 대비 압력 피처 (meas_p_w*_{mean,min,max}, meas_p_cmax, meas_p_m60 에서 p0 를 뺀 값)
  C_rel_p0   C_rel + p0 자체 (가설 5: 시작 압력 높은 실험은 상승폭 스케일이 작음 → 그룹별 임계값 학습)
  A_p0       A(압력 절대값 유지) + p0, 차이값(prel_*) 없음, 모든 행
  AC_p0      A_p0 + 과충전 행만 C (SOTA 구조에 p0 만 추가)
  A_aug      A 피처 + 압력 오프셋 증강 (실험마다 시작 압력을 U(0,75)로 옮긴 복제본 2개 추가)
  C_rel_w    C_rel + 시작 압력 높은 실험(p0>8) 가중치를 Test 비율(64%)에 맞춤  ※ Test 입력 분포 사용
  branch_p0  p0>8 이거나 과충전이면 C, 아니면 A
PRESS_VARIANTS=ref_AC,C_rel 처럼 일부만 실행 가능.

p0 = 실험 시작 30초 평균 (SOTA meas_p_rise 의 기준값, 인과적).
점수: macro F1 전체 / 과충전 / 시작 압력 높은 실험(hiP, p0>8) / 낮은 실험(loP).

  python press_exp.py sensor    # LightGBM fold 모델 → 확률 (CPU)
  python press_exp.py merge     # 결합·디코딩·점수
  python press_exp.py hyp       # 가설 5: 시작 압력 그룹별 상승폭 스케일 비교
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

SOTA = Path('/home/nxcolelxu/Team10-uni-dthon-6th-battery-thermal-runaway/sota_0.8812')
sys.path.insert(0, str(SOTA))
import features as FT  # noqa: E402
import model as SM  # noqa: E402

D = Path('/home/nxcolelxu/unid/data/battery_data_light')
OUT = Path(__file__).resolve().parent / 'press_exp'
(OUT / 'ckpt').mkdir(parents=True, exist_ok=True)
LAB = np.array([1, 2, 3, 4, 5, 6])
HI_P0 = 8.0
# 기존 ConvNeXt-tiny 확률: train 행 = CV OOF, val 행 = train 학습 모델 예측 (모두 해당 실험 미학습)
CNX = Path(__file__).resolve().parent / 'ft_convnext_tiny.npz'
TEST_HI_FRAC = 7 / 11
VARIANTS = ['ref_AC', 'C_all', 'C_rel', 'C_rel_p0', 'A_p0', 'AC_p0', 'A_aug', 'C_rel_w', 'branch_p0']
if os.environ.get('PRESS_VARIANTS'):
    VARIANTS = os.environ['PRESS_VARIANTS'].split(',')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def cap_rows(frame, m=500):
    parts = []
    for _, e in frame.groupby('experiment_id', sort=False):
        e = e.sort_values('relative_time_s')
        if len(e) > m:
            e = e.iloc[np.unique(np.round(np.linspace(0, len(e) - 1, m)).astype(int))]
        parts.append(e)
    return pd.concat(parts)


def frames():
    tr = pd.read_csv(D / 'train' / 'train.csv'); va = pd.read_csv(D / 'validation' / 'validation.csv')
    tr['_src'] = 'train'; va['_src'] = 'validation'
    df = pd.concat([tr, va], ignore_index=True)
    cap = cap_rows(df).reset_index(drop=True)
    exps = df.experiment_id.values
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(df, groups=exps))
    folds = []
    for k, (a, b) in enumerate(splits):
        held = set(exps[b])
        folds.append((k, a, np.where(cap.experiment_id.isin(held).values)[0]))
    # 실험 단위 p0 (점수 그룹용)
    d0 = df[df.relative_time_s <= 30].groupby('experiment_id').meas_p.mean()
    return df, cap, folds, d0


def rel_feats(X):
    p0 = X['meas_p'] - X['meas_p_rise']
    R = {}
    for w in (30, 120, 300):
        for s in ('mean', 'min', 'max'):
            R[f'prel_w{w}_{s}'] = X[f'meas_p_w{w}_{s}'] - p0
    R['prel_cmax'] = X['meas_p_cmax'] - p0
    R['prel_m60'] = X['meas_p_m60'] - p0
    return pd.DataFrame(R, index=X.index).astype('float32'), p0


def feats(frame, name):
    p = OUT / f'feat_{name}.pkl'
    if p.exists():
        return pd.read_pickle(p)
    X = FT.build(frame, 'rich')
    R, p0 = rel_feats(X)
    X = pd.concat([X, R], axis=1); X['_p0'] = p0.astype('float32')
    X.to_pickle(p)
    log(f'features {name}: {X.shape}')
    return X


def offset_copies(df, n=2, seed=0):
    rng = np.random.default_rng(seed); out = []
    p0 = df[df.relative_time_s <= 30].groupby('experiment_id').meas_p.mean()
    for c in range(n):
        d = df.copy()
        tgt = pd.Series(rng.uniform(0, 75, len(p0)), index=p0.index)
        off = (tgt - p0).reindex(d.experiment_id).values
        d['meas_p'] = pd.to_numeric(d.meas_p, errors='coerce') + off
        d['meas_p'] = d.meas_p.clip(lower=0)
        d['_orig_exp'] = d.experiment_id
        d['experiment_id'] = d.experiment_id + f'_aug{c}'
        d['sample_id'] = d.sample_id + f'_aug{c}'
        out.append(d)
    return pd.concat(out, ignore_index=True)


def sensor():
    import joblib, lightgbm as lgb
    df, cap, folds, d0 = frames()
    Xtr = feats(df, 'trval')                      # 학습: 전체 행
    Xev = feats(cap, 'trval_cap')                 # 평가: 실험당 500행 캡 (Test 조건)
    if 'A_aug' in VARIANTS:
        aug = offset_copies(df); Xaug = feats(aug, 'trval_aug'); y_aug = aug.label.values.astype(int)
    base = [c for c in Xtr.columns if not c.startswith('prel_') and c != '_p0']
    cols_c = SM.cols_without_abs_pressure(base)
    cols_rel = cols_c + [c for c in Xtr.columns if c.startswith('prel_')]
    log(f'A {len(base)} / C {len(cols_c)} / C_rel {len(cols_rel)}')
    y = df.label.values.astype(int)
    exp_hi = (d0 > HI_P0)
    params = dict(SM.LGB_PARAMS, n_jobs=6)

    def fit(name, k, X, yy, w=None):
        ck = OUT / 'ckpt' / f'lgbm_{name}_fold{k}.joblib'
        if ck.exists():
            return joblib.load(ck)
        m = lgb.LGBMClassifier(**params).fit(X, yy, sample_weight=w)
        joblib.dump(m, ck)
        return m

    for k, a, ev_idx in folds:
        ev = cap.iloc[ev_idx]; Xe = Xev.loc[ev.sample_id.values]
        t2 = ev.tr_method.values == 2
        hi_row = (Xe['_p0'].values > HI_P0)
        Xa = Xtr.iloc[a]; ya = y[a]
        train_exps = set(df.experiment_id.values[a])
        todo = [v for v in VARIANTS if not (OUT / f'sensor_{v}_fold{k}.npy').exists()]
        if not todo:
            continue
        mA = mC = mR = None
        if {'ref_AC', 'branch_p0'} & set(todo):
            mA = fit('A', k, Xa[base], ya); mC = fit('C', k, Xa[cols_c], ya)
        if 'C_all' in todo:
            mC = mC if mC is not None else fit('C', k, Xa[cols_c], ya)
        out = {}
        if 'ref_AC' in todo:
            P = mA.predict_proba(Xe[base]); P[t2] = mC.predict_proba(Xe[cols_c][t2]); out['ref_AC'] = P
        if 'C_all' in todo:
            out['C_all'] = mC.predict_proba(Xe[cols_c])
        if 'branch_p0' in todo:
            P = mA.predict_proba(Xe[base]); m = t2 | hi_row; P[m] = mC.predict_proba(Xe[cols_c][m]); out['branch_p0'] = P
        if 'C_rel' in todo:
            mR = fit('Crel', k, Xa[cols_rel], ya); out['C_rel'] = mR.predict_proba(Xe[cols_rel])
        if 'C_rel_p0' in todo:
            cp = cols_rel + ['_p0']
            out['C_rel_p0'] = fit('Crelp0', k, Xa[cp], ya).predict_proba(Xe[cp])
        if {'A_p0', 'AC_p0'} & set(todo):
            cp = base + ['_p0']
            P = fit('Ap0', k, Xa[cp], ya).predict_proba(Xe[cp])
            out['A_p0'] = P.copy()
            mC = mC if mC is not None else fit('C', k, Xa[cols_c], ya)
            P[t2] = mC.predict_proba(Xe[cols_c][t2]); out['AC_p0'] = P
        if 'C_rel_w' in todo:
            ea = df.experiment_id.values[a]; hi = exp_hi.reindex(ea).fillna(False).values
            f_hi = pd.Series(hi).groupby(ea).first().mean()
            w = np.where(hi, TEST_HI_FRAC / f_hi, (1 - TEST_HI_FRAC) / (1 - f_hi))
            out['C_rel_w'] = fit('Crelw', k, Xa[cols_rel], ya, w).predict_proba(Xe[cols_rel])
        if 'A_aug' in todo:
            am = aug._orig_exp.isin(train_exps).values
            XX = pd.concat([Xa[base], Xaug[base][am]]); yy = np.concatenate([ya, y_aug[am]])
            out['A_aug'] = fit('Aaug', k, XX, yy).predict_proba(Xe[base])
        for v, P in out.items():
            np.save(OUT / f'sensor_{v}_fold{k}.npy', P)
        log(f'fold{k} sensor done: {sorted(out)} (eval rows {len(ev)})')


def merge():
    df, cap, folds, d0 = frames()
    ev_all = np.concatenate([e for _, _, e in folds])
    fr = cap.iloc[ev_all].reset_index(drop=True)
    y = fr.label.values.astype(int)
    hi = fr.experiment_id.map(d0).values > HI_P0
    oc = fr.tr_method.values == 2
    c = np.load(CNX, allow_pickle=True)
    V = pd.DataFrame(c['probs'], index=c['sample_id'].astype(str)).loc[fr.sample_id.astype(str).values].to_numpy()
    groups = dict(all=np.ones(len(y), bool), overcharge=oc, hiP=hi, loP=~hi, hiP_tm1=hi & ~oc)
    log(f'rows {len(y)} | exps hiP {fr[hi].experiment_id.nunique()} loP {fr[~hi].experiment_id.nunique()} OC {fr[oc].experiment_id.nunique()}')
    res = {}
    for v in VARIANTS:
        if not all((OUT / f'sensor_{v}_fold{k}.npy').exists() for k, _, _ in folds):
            continue
        S = np.concatenate([np.load(OUT / f'sensor_{v}_fold{k}.npy') for k, _, _ in folds])
        for al in (0.1, 0.06):
            P = (1 - al) * S + al * V
            lab = SM.viterbi(fr, P)
            r = {g: float(f1_score(y[m], lab[m], labels=LAB, average='macro')) for g, m in groups.items()}
            r['per_class'] = f1_score(y, lab, labels=LAB, average=None).round(4).tolist()
            res[f'{v}@a{al}'] = r
            log(f'{v:10s} α={al:<4} all {r["all"]:.4f} | OC {r["overcharge"]:.4f} | hiP {r["hiP"]:.4f} (tm1 {r["hiP_tm1"]:.4f}) | loP {r["loP"]:.4f} | {r["per_class"]}')
    (OUT / 'results.json').write_text(json.dumps(res, indent=1))


def hyp():
    """가설 5: 시작 압력이 높은 실험은 상승폭 스케일도 다른가"""
    df, cap, folds, d0 = frames()
    rows = []
    for e, g in df.sort_values('relative_time_s').groupby('experiment_id'):
        p = pd.to_numeric(g.meas_p, errors='coerce'); p0 = d0[e]
        r = dict(exp=e, p0=p0, tm=g.tr_method.iloc[0], bat=g.bat_class.iloc[0], rise_max=p.max() - p0)
        for L in (3, 4, 6):
            on = g[g.label >= L]
            r[f'rise@L{L}'] = (pd.to_numeric(on.meas_p.iloc[0], errors='coerce') - p0) if len(on) else np.nan
        rows.append(r)
    E = pd.DataFrame(rows); E['grp'] = np.where(E.p0 > 60, 'p0>60', np.where(E.p0 > HI_P0, '8<p0<=60', 'p0<=8'))
    pd.set_option('display.width', 200)
    print(E.groupby(['tm', 'grp'])[['p0', 'rise_max', 'rise@L3', 'rise@L4', 'rise@L6']].agg(['count', 'median']).round(2).to_string())
    print('\nrise_max / p0 상관 (spearman):', round(E[['p0', 'rise_max']].corr('spearman').iloc[0, 1], 3))


if __name__ == '__main__':
    {'sensor': sensor, 'merge': merge, 'hyp': hyp}[sys.argv[1]]()
