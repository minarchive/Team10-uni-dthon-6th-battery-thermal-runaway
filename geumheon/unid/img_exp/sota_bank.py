"""SOTA 계열 센서 모델의 OOF 확률 은행 (앙상블용). final_oc4 와 같은 평가 틀:

  CV  : Train 78개 실험 GroupKFold(5, shuffle, seed 0) → 평가 fold 는 500행 캡(Test 조건)으로 피처를 다시 계산해 예측
  Val : Train 전체 학습 → Val 500행 캡
  Test: Train+Val 학습 → Test  (이 최종 모델은 blend_sota/weights/ 에 저장)

모델 (모두 SOTA 분기: 일반 실험 = A(전체 피처), 과충전 tr_method 2 = C(압력 절대값·v_peak_ratio 제외))
  lgb      : LightGBM, rich 166 피처 (SOTA 0.8812 그대로)
  et       : Extra Trees, rich 166 피처 (v3_sota_et0.8920 의 추가 모델)
  lgb_p23  : LightGBM, rich + 실험 기준 z-score + 2→3 전용 피처(압력 급등 시작·가열 가속)  ← 2→3 경계용 후보
  lgb_bal  : LightGBM, rich, class_weight='balanced'  ← 소수 단계(4·5·6) 다양성 후보

출력: sota_bank.npz — tr_ids/va_ids/te_ids, <모델>_cv / _val / _test (n×6)
"""
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

SOTA = Path('/home/nxcolelxu/Team10-uni-dthon-6th-battery-thermal-runaway/v3_sota_et0.8920')
sys.path.insert(0, str(SOTA))
import features as FT  # noqa: E402
import model as SM  # noqa: E402

D = Path('/home/nxcolelxu/unid/data/battery_data_light')
HERE = Path(__file__).resolve().parent
WDIR = HERE.parent / 'blend_sota' / 'weights'
WDIR.mkdir(parents=True, exist_ok=True)
CK = HERE / 'sota_bank_ckpt'; CK.mkdir(exist_ok=True)
NJ = int(os.environ.get('NJ', '8'))
ET_PARAMS = dict(n_estimators=500, min_samples_leaf=20, max_features=0.3, n_jobs=NJ, random_state=SM.SEED)
MODELS = {
    'lgb': dict(kind='lgb', mode='rich'),
    'et': dict(kind='et', mode='rich'),
    'lgb_p23': dict(kind='lgb', mode='rich_z_p23'),
    'lgb_bal': dict(kind='lgb', mode='rich', balanced=True),
}
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
    return pd.concat(parts).reset_index(drop=True)


def make(cfg):
    if cfg['kind'] == 'et':
        return ExtraTreesClassifier(**ET_PARAMS)
    import lightgbm as lgb
    p = dict(SM.LGB_PARAMS, n_jobs=NJ)
    if cfg.get('balanced'):
        p['class_weight'] = 'balanced'
    return lgb.LGBMClassifier(**p)


def arr(X, cfg):
    return X.values.astype(np.float32) if cfg['kind'] == 'et' else X


def fit_ac(name, tag, cfg, X, y, cols, cols_c, out_dir=CK):
    p = out_dir / f'{name}_{tag}.joblib'
    if p.exists():
        return joblib.load(p)
    M = dict(A=make(cfg).fit(arr(X[cols], cfg), y), C=make(cfg).fit(arr(X[cols_c], cfg), y), cols=cols, cols_c=cols_c, cfg=cfg)
    joblib.dump(M, p, compress=3)
    return M


def predict_ac(M, Xe, frame):
    cfg = M['cfg']
    P = np.zeros((len(Xe), 6)); t2 = frame.tr_method.values == 2
    cls = np.searchsorted(SM.LABELS, M['A'].classes_)
    P[:, cls] = M['A'].predict_proba(arr(Xe[M['cols']], cfg))
    if t2.any():
        Pc = np.zeros((int(t2.sum()), 6)); Pc[:, np.searchsorted(SM.LABELS, M['C'].classes_)] = M['C'].predict_proba(arr(Xe[M['cols_c']].iloc[np.where(t2)[0]], cfg))
        P[t2] = Pc
    return P


def main():
    tr = pd.read_csv(D / 'train' / 'train.csv'); va = pd.read_csv(D / 'validation' / 'validation.csv'); te = pd.read_csv(D / 'test' / 'test.csv')
    tr_cap, va_cap = cap_rows(tr), cap_rows(va)
    tv = pd.concat([tr, va], ignore_index=True)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    out = dict(tr_ids=tr_cap.sample_id.values.astype(str), va_ids=va_cap.sample_id.values.astype(str), te_ids=te.sample_id.values.astype(str))
    feats = {}

    def F(frame, nm, mode):
        if (nm, mode) not in feats:
            feats[(nm, mode)] = FT.build(frame, mode); log(f'features {nm}/{mode} {feats[(nm, mode)].shape}')
        return feats[(nm, mode)]

    for name, cfg in MODELS.items():
        mode = cfg['mode']
        Xtr = F(tr, 'tr', mode); cols = list(Xtr.columns); cols_c = SM.cols_without_abs_pressure(cols)
        Xtr_cap = F(tr_cap, 'tr_cap', mode)[cols]; Xva_cap = F(va_cap, 'va_cap', mode)[cols]
        ytr = tr.label.values.astype(int)
        P = np.zeros((len(tr_cap), 6))
        for k, (a, b) in enumerate(splits):
            M = fit_ac(name, f'cv_fold{k}', cfg, Xtr.iloc[a], ytr[a], cols, cols_c)
            m = ~tr_cap.experiment_id.isin(set(tr.experiment_id.iloc[a])).to_numpy()
            P[m] = predict_ac(M, Xtr_cap[m], tr_cap[m])
        out[f'{name}_cv'] = P
        M = fit_ac(name, 'val_train_all', cfg, Xtr, ytr, cols, cols_c)
        out[f'{name}_val'] = predict_ac(M, Xva_cap, va_cap)
        Xtv = F(tv, 'tv', mode)[cols]
        M = fit_ac(name, 'final_trval', cfg, Xtv, tv.label.values.astype(int), cols, cols_c, out_dir=WDIR)
        out[f'{name}_test'] = predict_ac(M, F(te, 'te', mode)[cols], te)
        log(f'{name} done (feats {len(cols)} / C {len(cols_c)})')
        np.savez(HERE / 'sota_bank.npz', **out)
    log('saved sota_bank.npz')


if __name__ == '__main__':
    main()
