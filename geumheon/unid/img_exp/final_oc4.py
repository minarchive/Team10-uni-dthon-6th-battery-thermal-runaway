"""45°C 규칙 + 2~6단계 전용 모델 (1단계 길이 제거). 이미지 = ConvNeXt-Tiny 파인튜닝 확률 0.1.

- 1단계: 실험별 t1 누적최대 < 45°C 인 행 = 1단계 (Train·Val 라벨과 100% 일치, 현재 시점까지만 사용 → 인과)
- 2~6단계 모델: 45°C 돌파 이후 행만으로 학습·예측
- 시계 재설정(reset): 시작 시각 기준 피처 3개(relative_time_s, oc_v_rise_rate, oc2_since45_over_t45) 제거,
  t_since45(45°C 돌파 이후 경과 초) 추가 → 1단계 길이가 입력에 들어가지 않음
설정
  R_J6wide        : 규칙 + 2~6 모델 + 시계 재설정, 증강 J6wide(과충전 흔들기 6개, w 0.3)
  R_J6wide_noreset: 규칙 + 2~6 모델, 피처는 그대로 (시계 재설정 효과 분리용)
  R_J2ref         : 규칙 + 2~6 모델 + 시계 재설정, 증강 J2ref
CV/Val/최종 구성과 체크포인트 저장은 final_oc3 와 동일.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import final_oc as O
from final_oc import F, load, D, score, log
import final_oc3 as R
from final_oc2 import fit_ckpt, cnn

W_IMG = 0.1
OUT = O.M.HERE / f'final_oc4_{os.environ.get("TAG", "gpu")}'
OUT.mkdir(exist_ok=True)
CV_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
ABS_TIME = ['relative_time_s', 'oc_v_rise_rate', 'oc2_since45_over_t45']
CFGS = {
    'R_J6wide':         dict(aug='wide', w=0.3, reset=True),
    'R_J6wide_noreset': dict(aug='wide', w=0.3, reset=False),
    'R_J2ref':          dict(aug='ref', w=0.3, reset=True),
}
if os.environ.get('OC4_CFGS'):
    CFGS = {k: CFGS[k] for k in os.environ['OC4_CFGS'].split(',')}
LAB = np.array(F.LABELS)


def t45_info(fr):
    """행별 (45°C 돌파 여부, 돌파 이후 경과 초). 실험별 시간순 t1 누적최대 기준."""
    t = fr.relative_time_s.to_numpy(float)
    t1 = fr.meas_t1.to_numpy(float)
    hit = np.zeros(len(fr), bool); since = np.full(len(fr), np.nan)
    for _, idx in fr.reset_index(drop=True).groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        h = np.fmax.accumulate(np.where(np.isfinite(t1[idx]), t1[idx], -np.inf)) >= 45
        hit[idx] = h
        if h.any():
            f = int(np.argmax(h)); since[idx[f:]] = t[idx[f:]] - t[idx[f]]
    return hit, since


def feats_r(fr):
    X = R.feats(fr)
    hit, since = t45_info(fr)
    X['t_since45'] = since.astype(np.float32)
    return X, pd.Series(hit, index=fr.sample_id.values)


def predict(model_path, s, X, y, w, Sq, Hq, ids, cols):
    """45°C 이전 행 = 1단계(확률 1), 이후 행 = 2~6단계 모델 확률."""
    P = np.zeros((len(ids), 6)); h = Hq.loc[ids].to_numpy()
    P[~h, 0] = 1.0
    if h.any():
        m = fit_ckpt(model_path, s, X, y, w)
        P[np.ix_(h, np.searchsorted(LAB, m.classes_))] = m.predict_proba(Sq.loc[ids[h], cols])
    return P


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    for f in (tr, va, te):
        f['sample_id'] = f.sample_id.astype(str)
    tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)
    need = sorted({c['aug'] for c in CFGS.values()})
    aug = {k: (R.make_aug3(tr, 0, R.AUGS[k]), R.make_aug3(va, 1, R.AUGS[k])) for k in need}
    log('aug rows', {k: (len(a), len(b)) for k, (a, b) in aug.items()})
    S, H = {}, {}
    for nm, f in (('tr', tr), ('tr_cap', tr_cap), ('va', va), ('va_cap', va_cap), ('te', te)):
        S[nm], H[nm] = feats_r(f)
    SA, HA = {}, {}
    for k, (atr, ava) in aug.items():
        for s, a in (('tr', atr), ('va', ava)):
            SA[(k, s)], HA[(k, s)] = feats_r(a)
    all_cols = list(S['tr'].columns)
    log(f'feats {len(all_cols)}; train rows after 45C {int(H["tr"].sum())}/{len(tr)}')
    # 규칙 검증 (Train·Val 라벨)
    for nm, f in (('tr', tr), ('va', va)):
        ok = ((f.label.astype(int).to_numpy() == 1) == ~H[nm].loc[f.sample_id.values].to_numpy()).mean()
        log(f'45C rule == (label==1) on {nm}: {ok:.4f}')

    def cols_of(cfg):
        return [c for c in all_cols if not (cfg['reset'] and c in ABS_TIME)] if cfg['reset'] else [c for c in all_cols if c != 't_since45']

    def train_set(Sf, Hf, Sa, Ha, fr_rows, af, cfg):
        cols = cols_of(cfg)
        ids = fr_rows.sample_id.values[Hf.loc[fr_rows.sample_id.values].to_numpy()]
        aids = af.sample_id.values[Ha.loc[af.sample_id.values].to_numpy()]
        lab = pd.concat([fr_rows.set_index('sample_id').label, af.set_index('sample_id').label]).astype(int)
        X = pd.concat([Sf.loc[ids, cols], Sa.loc[aids, cols]])
        y = np.r_[lab.loc[ids].to_numpy(), lab.loc[aids].to_numpy()]
        w = np.r_[np.ones(len(ids)), np.full(len(aids), cfg['w'])]
        return X, y, w, cols

    res_path = OUT / 'results.json'
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    c_tr = cnn('ft_convnext_tiny.npz', tr_cap.sample_id.values)
    c_va = cnn('ft_convnext_tiny.npz', va_cap.sample_id.values)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    for nm, cfg in CFGS.items():
        if nm in res and 'val' in res[nm]:
            log(f'skip {nm}'); continue
        atr = aug[cfg['aug']][0]; Sa, Ha = SA[(cfg['aug'], 'tr')], HA[(cfg['aug'], 'tr')]
        P = np.zeros((len(tr_cap), 6))
        for s in CV_SEEDS:
            for k, (a, b) in enumerate(splits):
                fa = tr.iloc[a]; ea = set(fa.experiment_id); af = atr[atr.src_exp.isin(ea)]
                X, y, w, cols = train_set(S['tr'], H['tr'], Sa, Ha, fa, af, cfg)
                m = ~tr_cap.experiment_id.isin(ea).to_numpy()
                P[m] += predict(OUT / 'ckpt' / nm / f'cv_fold{k}_seed{s}.joblib', s, X, y, w,
                                S['tr_cap'], H['tr_cap'], tr_cap.sample_id.values[m], cols) / len(CV_SEEDS)
        np.save(OUT / f'oof_{nm}.npy', P)
        cv, cv_oc, pc = score(tr_cap, (1 - W_IMG) * P + W_IMG * c_tr)
        X, y, w, cols = train_set(S['tr'], H['tr'], Sa, Ha, tr, atr, cfg)
        pv = sum(predict(OUT / 'ckpt' / nm / f'val_seed{s}.joblib', s, X, y, w, S['va_cap'], H['va_cap'],
                         va_cap.sample_id.values, cols) for s in CV_SEEDS) / len(CV_SEEDS)
        np.save(OUT / f'valprobs_{nm}.npy', pv)
        v, v_oc, vpc = score(va_cap, (1 - W_IMG) * pv + W_IMG * c_va)
        res[nm] = dict(cv=cv, cv_oc=cv_oc, cv_per_class=pc, val=v, val_oc=v_oc, val_per_class=vpc, n_feat=len(cols))
        res_path.write_text(json.dumps(res, indent=1))
        log(f'{nm:17s} +CNX0.1: CV {cv:.4f} OC {cv_oc:.4f} | Val {v:.4f} OC {v_oc:.4f} | CV {pc} | Val {vpc}')

    trva = pd.concat([tr, va], ignore_index=True)
    Stv, Htv = pd.concat([S['tr'], S['va']]), pd.concat([H['tr'], H['va']])
    c_te = cnn('ft_convnext_tiny_trval.npz', te.sample_id.values)
    ss = pd.read_csv(D / 'sample_submission.csv', dtype={'sample_id': str})
    import joblib
    for nm, cfg in CFGS.items():
        atv = pd.concat(aug[cfg['aug']], ignore_index=True)
        Satv = pd.concat([SA[(cfg['aug'], 'tr')], SA[(cfg['aug'], 'va')]])
        Hatv = pd.concat([HA[(cfg['aug'], 'tr')], HA[(cfg['aug'], 'va')]])
        X, y, w, cols = train_set(Stv, Htv, Satv, Hatv, trva, atv, cfg)
        pS = sum(predict(OUT / 'ckpt' / nm / f'final_seed{s}.joblib', s, X, y, w, S['te'], H['te'],
                         te.sample_id.values, cols) for s in FINAL_SEEDS) / len(FINAL_SEEDS)
        joblib.dump(dict(cols=cols, cfg=cfg, aug=R.AUGS[cfg['aug']], rule='t1 cummax < 45C -> stage 1',
                         img_w=W_IMG, img_model='ft_convnext_tiny_trval_models', tau=0.7), OUT / 'ckpt' / nm / 'meta.joblib')
        np.save(OUT / f'test_probs_{nm}.npy', pS)
        lab = F.postprocess(te, (1 - W_IMG) * pS + W_IMG * c_te, 0.7)
        sub = ss[['sample_id']].merge(pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab}), on='sample_id', how='left')
        assert sub.label.notna().all() and len(sub) == len(ss)
        p = OUT / f'submission_{nm}_convnext0.1.csv'
        sub.to_csv(p, index=False)
        log(f'wrote {p.name} dist {sub.label.value_counts().sort_index().to_dict()}')
    log('ALL DONE')


if __name__ == '__main__':
    main()
