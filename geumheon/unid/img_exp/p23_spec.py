"""2↔3 전용 이진 판별 모델: 45°C 돌파 이후 행에서 P(단계 ≥ 3) 만 학습 (앙상블의 2/3 확률 배분 보정용).

피처: SOTA rich + 실험 기준 z-score + 2→3 전용 피처(압력 급등 시작·가열 가속) = rich_z_p23
분기: 일반 실험 A(전체 피처), 과충전 C(압력 절대값·v_peak_ratio 제외) — sota_bank 와 같음
모델: LightGBM(SOTA 설정, seed 3개 평균) · Extra Trees
평가 틀: sota_bank 와 동일 (Train 5-fold OOF 500행 캡 / Train 전체→Val / Train+Val→Test, 최종 가중치는 blend_sota/weights/)
출력: p23_spec.npz — tr_ids/va_ids/te_ids, <모델>_cv/_val/_test = P(≥3) (45°C 이전 행은 0)
"""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
import sota_bank as B
from sota_bank import FT, SM, D, HERE, WDIR, CK, NJ, log, cap_rows

MODE = 'rich_z_p23'


def after45(frame):
    t = frame.relative_time_s.to_numpy(float); x = pd.to_numeric(frame.meas_t1, errors='coerce').to_numpy(float); out = np.zeros(len(frame), bool)
    for _, idx in frame.reset_index(drop=True).groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]; out[idx] = np.fmax.accumulate(np.nan_to_num(x[idx], nan=-np.inf)) >= 45
    return out


def make(kind, seed):
    if kind == 'et':
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=20, max_features=0.3, n_jobs=NJ, random_state=seed)
    import lightgbm as lgb
    return lgb.LGBMClassifier(**dict(SM.LGB_PARAMS, n_jobs=NJ, random_state=seed))


def fit(kind, tag, X, b, cols, cols_c, out_dir=CK):
    p = out_dir / f'p23spec_{kind}_{tag}.joblib'
    if p.exists():
        return joblib.load(p)
    seeds = (42, 7, 1) if kind == 'lgb' else (42,)
    v = (lambda Z: Z.values.astype(np.float32)) if kind == 'et' else (lambda Z: Z)
    M = dict(kind=kind, cols=cols, cols_c=cols_c, A=[make(kind, s).fit(v(X[cols]), b) for s in seeds], C=[make(kind, s).fit(v(X[cols_c]), b) for s in seeds])
    joblib.dump(M, p, compress=3); return M


def predict(M, Xe, frame, h):
    v = (lambda Z: Z.values.astype(np.float32)) if M['kind'] == 'et' else (lambda Z: Z)
    q = np.zeros(len(Xe)); t2 = frame.tr_method.values == 2
    for mask, key, cols in ((h & ~t2, 'A', M['cols']), (h & t2, 'C', M['cols_c'])):
        if mask.any():
            Z = v(Xe[cols].iloc[np.where(mask)[0]]); q[mask] = np.mean([m.predict_proba(Z)[:, 1] for m in M[key]], 0)
    return q


def main():
    tr = pd.read_csv(D / 'train' / 'train.csv'); va = pd.read_csv(D / 'validation' / 'validation.csv'); te = pd.read_csv(D / 'test' / 'test.csv')
    tr_cap, va_cap = cap_rows(tr), cap_rows(va); tv = pd.concat([tr, va], ignore_index=True)
    out = dict(tr_ids=tr_cap.sample_id.values.astype(str), va_ids=va_cap.sample_id.values.astype(str), te_ids=te.sample_id.values.astype(str))
    X = {nm: FT.build(f, MODE) for nm, f in (('tr', tr), ('tr_cap', tr_cap), ('va_cap', va_cap), ('tv', tv), ('te', te))}; log('features done')
    H = {nm: after45(f) for nm, f in (('tr', tr), ('tr_cap', tr_cap), ('va_cap', va_cap), ('tv', tv), ('te', te))}
    cols = list(X['tr'].columns); cols_c = SM.cols_without_abs_pressure(cols)
    btr = (tr.label.values >= 3).astype(int); btv = (tv.label.values >= 3).astype(int)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    for kind in ('lgb', 'et'):
        q = np.zeros(len(tr_cap))
        for k, (a, b_) in enumerate(splits):
            a = a[H['tr'][a]]
            M = fit(kind, f'cv_fold{k}', X['tr'].iloc[a], btr[a], cols, cols_c)
            m = ~tr_cap.experiment_id.isin(set(tr.experiment_id.iloc[a])).to_numpy()
            q[m] = predict(M, X['tr_cap'][m], tr_cap[m], H['tr_cap'][m])
        out[f'{kind}_cv'] = q
        a = np.where(H['tr'])[0]; M = fit(kind, 'val_train_all', X['tr'].iloc[a], btr[a], cols, cols_c)
        out[f'{kind}_val'] = predict(M, X['va_cap'], va_cap, H['va_cap'])
        a = np.where(H['tv'])[0]; M = fit(kind, 'final_trval', X['tv'].iloc[a], btv[a], cols, cols_c, out_dir=WDIR)
        out[f'{kind}_test'] = predict(M, X['te'], te, H['te'])
        from sklearn.metrics import roc_auc_score
        y3 = (tr_cap.label.values >= 3).astype(int); hh = H['tr_cap']
        log(f'{kind}: CV AUC(45°C 이후) {roc_auc_score(y3[hh], q[hh]):.4f}  acc@0.5 {((q[hh] > 0.5) == y3[hh]).mean():.4f}')
        np.savez(HERE / 'p23_spec.npz', **out)
    log('saved p23_spec.npz')


if __name__ == '__main__':
    main()
