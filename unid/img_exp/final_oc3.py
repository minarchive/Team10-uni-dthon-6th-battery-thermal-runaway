"""과충전 공격적 증강 3차 + 과충전 전용 모델. 이미지 = ConvNeXt-Tiny 파인튜닝 확률 0.1.

센서 피처: v3 + 궤적 A + OC 11 + OC2 23 + 셀제품×과충전 조합 20 (2차 최고 과충전 설정과 동일)
증강(과충전 실험만, tr_method=2):
  base  : a0(원본 캡) + a1,a2(L1 ×1.2~1.6)
  jit   : J개 변형 = L1 ×1.0~1.6 → [2·3단계 각각 ×r23] → 4·5단계 ×r45 → 행 솎기(n행) → [센서 값 흔들기]
설정 (OC3_CFGS 로 일부만 실행 가능)
  J2ref   : 2차 S_oc2_j_w03 재현 (J=2, r45 0.7~1.4, r23 없음, 500행, 노이즈 없음, w 0.3)
  J6wide  : J=6, r45 0.5~2.0, r23 0.7~1.5, 500행, w 0.3
  J6thin  : J6wide + 행 솎기 300~500 + 센서 노이즈, w 0.3
  J6thin_w6 : J6thin, w 0.6
  J6thin_ex50 : J6thin + 과충전 전용 모델(과충전 원본+증강, w 1) 을 과충전 행에 50% 결합
  J6thin_ex100: 과충전 행은 전용 모델만
최종 확률 = 0.9·센서 + 0.1·ConvNeXt → 확신 cummax τ=0.7
모든 모델(CV fold, Val, 최종, 전용 모델)은 체크포인트 저장 + 있으면 재사용. 모든 설정의 Test 확률·제출 파일 저장.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import final_oc as O
from final_oc import F, load, D, sensor_feats, stretch_l1, cap_idx, score, log
import final_oc2 as Q
from final_oc2 import feat_oc2, feat_combo, stretch_seg, fit_ckpt, cnn

W_IMG = 0.1
OUT = O.M.HERE / f'final_oc3_{os.environ.get("TAG", "gpu")}'
OUT.mkdir(exist_ok=True)
CV_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)

AUGS = {
    'ref':  dict(J=2, r45=(0.7, 1.4), r23=None, thin=None, noise=False),
    'wide': dict(J=6, r45=(0.5, 2.0), r23=(0.7, 1.5), thin=None, noise=False),
    'thin': dict(J=6, r45=(0.5, 2.0), r23=(0.7, 1.5), thin=(300, 500), noise=True),
    'thin2': dict(J=6, r45=(0.5, 2.0), r23=(0.7, 1.5), thin=(300, 500), noise=True, noise_temp=False),
    'thin3': dict(J=6, r45=(0.5, 2.0), r23=(0.7, 1.5), thin=(300, 500), noise=True, noise_temp=False, v_scale=0.005),
    # 4차: 더 공격적
    'w12':  dict(J=12, r45=(0.5, 2.0), r23=(0.7, 1.5), thin=None, noise=False),
    'w20':  dict(J=20, r45=(0.5, 2.0), r23=(0.7, 1.5), thin=None, noise=False),
    'x12':  dict(J=12, r45=(0.33, 3.0), r23=(0.5, 2.0), thin=None, noise=False, r1=(1.0, 2.0)),
}
CFGS = {
    'J2ref':        dict(aug='ref', w=0.3, ex=0.0),
    'J6wide':       dict(aug='wide', w=0.3, ex=0.0),
    'J6thin':       dict(aug='thin', w=0.3, ex=0.0),
    'J6thin_w6':    dict(aug='thin', w=0.6, ex=0.0),
    'J6thin_ex50':  dict(aug='thin', w=0.3, ex=0.5),
    'J6thin_ex100': dict(aug='thin', w=0.3, ex=1.0),
    'J6thin2':       dict(aug='thin2', w=0.3, ex=0.0),
    'J6thin2_ex50':  dict(aug='thin2', w=0.3, ex=0.5),
    'J6thin2_ex100': dict(aug='thin2', w=0.3, ex=1.0),
    'J6thin3':       dict(aug='thin3', w=0.3, ex=0.0),
    'J6thin3_ex50':  dict(aug='thin3', w=0.3, ex=0.5),
    'J12wide':      dict(aug='w12', w=0.3, ex=0.0),
    'J20wide':      dict(aug='w20', w=0.3, ex=0.0),
    'J20wide_w15':  dict(aug='w20', w=0.15, ex=0.0),
    'J12xwide':     dict(aug='x12', w=0.3, ex=0.0),
}
if os.environ.get('OC3_CFGS'):
    CFGS = {k: CFGS[k] for k in os.environ['OC3_CFGS'].split(',')}


def noise_exp(v, rng, temp=True, v_scale=0.03):
    v = v.copy()
    v['meas_v'] = v.meas_v * rng.uniform(1 - v_scale, 1 + v_scale)
    off = rng.uniform(-2, 2)
    if temp:                                   # 온도 흔들기는 1→2 규칙(t1 45°C)을 깨므로 thin2 에서는 끔
        for c in ('meas_t1', 'meas_t2', 'thermal_t'):
            v[c] = v[c] + off
    for c in F.GAS:
        v[c] = v[c] * rng.uniform(0.7, 1.4)
    v['meas_p'] = v.meas_p * rng.uniform(0.95, 1.05)
    v['meas_t1_plus_t2'] = v.meas_t1 + v.meas_t2
    return v


def make_aug3(fr, seed, spec):
    rng = np.random.default_rng(seed)
    parts = []
    for eid, e in fr[fr.tr_method == 2].groupby('experiment_id'):
        vs = [('a0', e.sort_values('relative_time_s'))] if len(e) > F.MAX_ROWS else []
        vs += [(f'a{k + 1}', stretch_l1(e, rng.uniform(1.2, 1.6))) for k in range(2)]
        for k in range(spec['J']):
            v = stretch_l1(e, rng.uniform(*spec.get('r1', (1.0, 1.6))))
            if spec['r23']:
                v = stretch_seg(v, [2], rng.uniform(*spec['r23']))
                v = stretch_seg(v, [3], rng.uniform(*spec['r23']))
            v = stretch_seg(v, [4, 5], rng.uniform(*spec['r45']))
            if spec['noise']:
                v = noise_exp(v, rng, temp=spec.get('noise_temp', True), v_scale=spec.get('v_scale', 0.03))
            vs.append((f'j{k}', v))
        for tag, v in vs:
            m = int(rng.integers(spec['thin'][0], spec['thin'][1] + 1)) if (spec['thin'] and tag.startswith('j')) else F.MAX_ROWS
            v = v.sort_values('relative_time_s')
            v = v.iloc[cap_idx(len(v), m)].copy()
            v['src_exp'] = eid; v['experiment_id'] = f'{eid}#{tag}'
            v['sample_id'] = [f'{eid}#{tag}_{i}' for i in range(len(v))]
            parts.append(v)
    return pd.concat(parts, ignore_index=True)


def feats(f):
    return pd.concat([sensor_feats(f), feat_oc2(f), feat_combo(f)], axis=1)


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    for f in (tr, va, te):
        f['sample_id'] = f.sample_id.astype(str)
    tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)
    need = sorted({c['aug'] for c in CFGS.values()})
    aug = {k: (make_aug3(tr, 0, AUGS[k]), make_aug3(va, 1, AUGS[k])) for k in need}
    log('aug rows', {k: (len(a), len(b)) for k, (a, b) in aug.items()})
    S = {nm: feats(f) for nm, f in (('tr', tr), ('tr_cap', tr_cap), ('va', va), ('va_cap', va_cap), ('te', te))}
    SA = {(k, s): feats(a) for k, (atr, ava) in aug.items() for s, a in (('tr', atr), ('va', ava))}
    cols = list(S['tr'].columns)
    log(f'feats {len(cols)}')
    ytr = tr.label.astype(int).to_numpy()
    oc_tr = (tr.tr_method == 2).to_numpy()

    def train_set(Sf, Sa, fr_rows, af, w_aug, oc_only=False):
        ids = fr_rows.sample_id.values
        if oc_only:
            ids = ids[(fr_rows.tr_method == 2).to_numpy()]
        X = pd.concat([Sf.loc[ids, cols], Sa.loc[af.sample_id.values, cols]])
        y = np.r_[Sf.loc[ids].index.map(dict(zip(fr_rows.sample_id, fr_rows.label.astype(int)))).to_numpy(int),
                  af.label.astype(int).to_numpy()]
        w = np.r_[np.ones(len(ids)), np.full(len(af), w_aug)]
        return X, y, w

    def predict_cfg(nm, cfg, tag, s, Xg, yg, wg, Xe, ye, we, Xq, oc_q):
        p = fit_ckpt(OUT / 'ckpt' / nm / f'{tag}_seed{s}.joblib', s, Xg, yg, wg).predict_proba(Xq)
        if cfg['ex'] > 0 and oc_q.any():
            ex = fit_ckpt(OUT / 'ckpt' / f'expert_{cfg["aug"]}' / f'{tag}_seed{s}.joblib', s, Xe, ye, we)
            pe = np.zeros_like(p)
            pe[:, np.searchsorted(np.array(F.LABELS), ex.classes_)] = ex.predict_proba(Xq[oc_q])
            p[oc_q] = (1 - cfg['ex']) * p[oc_q] + cfg['ex'] * pe
        return p

    res_path = OUT / 'results.json'
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    c_tr = cnn('ft_convnext_tiny.npz', tr_cap.sample_id.values)
    c_va = cnn('ft_convnext_tiny.npz', va_cap.sample_id.values)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    for nm, cfg in CFGS.items():
        if nm in res and 'val' in res[nm]:
            log(f'skip {nm}'); continue
        atr = aug[cfg['aug']][0]; Sa = SA[(cfg['aug'], 'tr')]
        P = np.zeros((len(tr_cap), 6))
        for s in CV_SEEDS:
            for k, (a, b) in enumerate(splits):
                fa = tr.iloc[a]; ea = set(fa.experiment_id); af = atr[atr.src_exp.isin(ea)]
                Xg, yg, wg = train_set(S['tr'], Sa, fa, af, cfg['w'])
                Xe, ye, we = train_set(S['tr'], Sa, fa, af, 1.0, oc_only=True) if cfg['ex'] else (None, None, None)
                m = ~tr_cap.experiment_id.isin(ea).to_numpy()
                q = tr_cap[m]
                P[m] += predict_cfg(nm, cfg, f'cv_fold{k}', s, Xg, yg, wg, Xe, ye, we,
                                    S['tr_cap'].loc[q.sample_id.values, cols], (q.tr_method == 2).to_numpy()) / len(CV_SEEDS)
        cv, cv_oc, pc = score(tr_cap, (1 - W_IMG) * P + W_IMG * c_tr)
        np.save(OUT / f'oof_{nm}.npy', P)
        Xg, yg, wg = train_set(S['tr'], Sa, tr, atr, cfg['w'])
        Xe, ye, we = train_set(S['tr'], Sa, tr, atr, 1.0, oc_only=True) if cfg['ex'] else (None, None, None)
        pv = sum(predict_cfg(nm, cfg, 'val', s, Xg, yg, wg, Xe, ye, we, S['va_cap'].loc[va_cap.sample_id.values, cols],
                             (va_cap.tr_method == 2).to_numpy()) for s in CV_SEEDS) / len(CV_SEEDS)
        np.save(OUT / f'valprobs_{nm}.npy', pv)
        v, v_oc, vpc = score(va_cap, (1 - W_IMG) * pv + W_IMG * c_va)
        res[nm] = dict(cv=cv, cv_oc=cv_oc, cv_per_class=pc, val=v, val_oc=v_oc, val_per_class=vpc)
        res_path.write_text(json.dumps(res, indent=1))
        log(f'{nm:13s} +CNX0.1: CV {cv:.4f} OC {cv_oc:.4f} | Val {v:.4f} OC {v_oc:.4f} | CV {pc} | Val {vpc}')

    # 최종: 모든 설정 (Train+Val, 5시드) → 제출
    trva = pd.concat([tr, va], ignore_index=True)
    Stv = pd.concat([S['tr'], S['va']])
    c_te = cnn('ft_convnext_tiny_trval.npz', te.sample_id.values)
    ss = pd.read_csv(D / 'sample_submission.csv', dtype={'sample_id': str})
    Xte = S['te'].loc[te.sample_id.values, cols]; oc_te = (te.tr_method == 2).to_numpy()
    import joblib
    for nm, cfg in CFGS.items():
        p_sub = OUT / f'submission_{nm}_convnext0.1.csv'
        atv = pd.concat(aug[cfg['aug']], ignore_index=True)
        Satv = pd.concat([SA[(cfg['aug'], 'tr')], SA[(cfg['aug'], 'va')]])
        Xg, yg, wg = train_set(Stv, Satv, trva, atv, cfg['w'])
        Xe, ye, we = train_set(Stv, Satv, trva, atv, 1.0, oc_only=True) if cfg['ex'] else (None, None, None)
        pS = sum(predict_cfg(nm, cfg, 'final', s, Xg, yg, wg, Xe, ye, we, Xte, oc_te) for s in FINAL_SEEDS) / len(FINAL_SEEDS)
        joblib.dump(dict(cols=cols, cfg=cfg, aug=AUGS[cfg['aug']], img_w=W_IMG, img_model='ft_convnext_tiny_trval_models',
                         tau=0.7, ckpt_dir=str(OUT / 'ckpt' / nm)), OUT / 'ckpt' / nm / 'meta.joblib')
        np.save(OUT / f'test_probs_{nm}.npy', pS)
        lab = F.postprocess(te, (1 - W_IMG) * pS + W_IMG * c_te, 0.7)
        sub = ss[['sample_id']].merge(pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab}), on='sample_id', how='left')
        assert sub.label.notna().all() and len(sub) == len(ss)
        sub.to_csv(p_sub, index=False)
        log(f'wrote {p_sub.name} dist {sub.label.value_counts().sort_index().to_dict()}')
    log('ALL DONE')


if __name__ == '__main__':
    main()
