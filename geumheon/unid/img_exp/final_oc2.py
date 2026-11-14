"""과충전 보강 2차: 2→3 경계 피처 + 증강 가중치 + 4·5단계 구간 흔들기. 이미지 = ConvNeXt-Tiny 파인튜닝 확률 0.1.

센서 설정 (모두 v3 + 궤적 A + 과충전 OC 피처 11개 기반)
  A  S_oc_aug        : 기존 (L1 늘리기 ×2 + 캡, 증강 가중치 1)
  B  S_oc_aug_w03    : 증강 행 가중치 0.3
  C  S_oc2_w03       : + 2→3 경계 피처(OC2) , 증강 가중치 0.3
  D  S_oc2_j_w03     : C + 4·5단계 구간 시간 흔들기(×0.7~1.4) 변형 2개
  E  S_oc2_j_w1      : D 와 같고 증강 가중치 1
최종 확률 = 0.9·센서 + 0.1·ConvNeXt  → 확신 cummax τ=0.7
CV: Train GroupKFold(5, seed 0) × 3시드 (ConvNeXt 는 같은 fold OOF). 선택: CV 전체, 동률(±0.0005)이면 과충전 CV.
Val: Train 학습 + ConvNeXt Train-fold 모델 평균. 최종: Train+Val 5시드 + ConvNeXt Train+Val 파인튜닝.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import final_oc as O
from final_oc import F, load, D, sensor_feats, stretch_l1, cap_idx, score, log
from exp_ctx import exp_groups, since_first
import exp_img

W_IMG = 0.1
OUT = O.M.HERE / f'final_oc2_{os.environ.get("TAG", "gpu")}'
OUT.mkdir(exist_ok=True)
CV_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# 2→3 경계 피처 (인과)
# ---------------------------------------------------------------------------
REL_SIG = {'gas_sum': (1.5, 3, 10), 'meas_co': (1.5, 3, 10), 'meas_co2': (1.2, 2, 5), 'meas_p': (1.05, 1.2, 1.5)}


def feat_oc2(fr):
    t, groups = exp_groups(fr)
    n = len(fr)
    d = fr.copy()
    d['gas_sum'] = d[F.GAS].sum(1)
    v = d.meas_v.to_numpy(float); t1 = d.meas_t1.to_numpy(float)
    cols = {k: np.full(n, np.nan) for k in ('oc2_since_vpk', 'oc2_t1rate_since_vpk', 'oc2_since45_over_t45',
                                             'oc2_t1_over_t1vpk', 'oc2_vslope30_rel', 'oc2_since_vdrop0.2',
                                             'oc2_since_vdrop0.5')}
    for idx in groups:
        tt, vv, tt1 = t[idx], v[idx], t1[idx]
        cm = np.fmax.accumulate(np.where(np.isfinite(vv), vv, -np.inf))
        pk = np.maximum.accumulate(np.where(np.isfinite(vv) & (vv >= cm), np.arange(len(idx)), 0))
        since = tt - tt[pk]
        cols['oc2_since_vpk'][idx] = since
        cols['oc2_t1rate_since_vpk'][idx] = (tt1 - tt1[pk]) / np.maximum(since, 5.0)
        cols['oc2_t1_over_t1vpk'][idx] = tt1 / np.maximum(tt1[pk], 1.0)
        hit = np.fmax.accumulate(np.where(np.isfinite(tt1), tt1, -np.inf)) >= 45
        if hit.any():
            f = int(np.argmax(hit))
            cols['oc2_since45_over_t45'][idx[f:]] = (tt[f:] - tt[f]) / max(tt[f], 5.0)
        j = np.clip(np.searchsorted(tt, tt - 30, side='right') - 1, 0, None)
        sl = (vv - vv[j]) / np.maximum(tt - tt[j], 5.0)
        cols['oc2_vslope30_rel'][idx] = sl / np.maximum(np.fmax.accumulate(np.nan_to_num(np.abs(sl))), 1e-4)
        drop = cm - vv
        for th in (0.2, 0.5):
            h = np.nan_to_num(drop, nan=-1) >= th
            o = np.full(len(idx), np.nan)
            if h.any():
                f = int(np.argmax(h)); o[f:] = tt[f:] - tt[f]
            cols[f'oc2_since_vdrop{th}'][idx] = o
    # 가스·압력: 처음 60초 평균 대비 배수 최초 도달 이후 경과 시간 (60초 전에는 지금까지 평균)
    for s, ks in REL_SIG.items():
        x = d[s].to_numpy(float)
        base = np.full(n, np.nan)
        for idx in groups:
            xx, tt = x[idx], t[idx]
            m = np.isfinite(xx)
            cs, cn = np.cumsum(np.where(m, xx, 0)), np.cumsum(m)
            k60 = np.searchsorted(tt, 60, side='right') - 1
            b = cs / np.maximum(cn, 1)
            if k60 >= 0:
                b[k60:] = b[k60]
            base[idx] = b
        ratio = x / np.maximum(np.abs(base), 1e-3)
        cols[f'oc2_{s}_ratio'] = ratio
        for k in ks:
            cols[f'oc2_since_{s}_x{k}'] = since_first(t, groups, np.nan_to_num(ratio, nan=-np.inf) >= k)
    return pd.DataFrame(cols, index=fr.sample_id.values).astype(np.float32)


# ---------------------------------------------------------------------------
# 증강
# ---------------------------------------------------------------------------
def stretch_seg(e, labels, s):
    """labels 구간의 시간축만 s배 (5초 격자, 직전 관측 유지), 이후 행은 평행이동."""
    e = e.sort_values('relative_time_s')
    t = e.relative_time_s.to_numpy(float)
    m = np.isin(e.label.to_numpy(), labels)
    if not m.any():
        return e
    a, b = t[m].min(), t[m].max() + 5.0
    pre = e[t < a]
    new_t = np.arange(a, a + (b - a) * s, 5.0)
    src = np.searchsorted(t, a + (new_t - a) / s, side='right') - 1
    mid = e.iloc[np.clip(src, 0, None)].copy(); mid['relative_time_s'] = new_t
    post = e[t >= b].copy(); post['relative_time_s'] = post.relative_time_s + (b - a) * (s - 1)
    return pd.concat([pre, mid, post])


def make_aug2(fr, seed, jitter):
    rng = np.random.default_rng(seed)
    parts = []
    for eid, e in fr[fr.tr_method == 2].groupby('experiment_id'):
        vs = [('a0', e.sort_values('relative_time_s'))] if len(e) > F.MAX_ROWS else []
        vs += [(f'a{k + 1}', stretch_l1(e, rng.uniform(1.2, 1.6))) for k in range(2)]
        if jitter:
            vs += [(f'j{k}', stretch_seg(stretch_l1(e, rng.uniform(1.0, 1.5)), [4, 5], rng.uniform(0.7, 1.4))) for k in range(2)]
        for tag, v in vs:
            v = v.iloc[cap_idx(len(v))].copy()
            v['src_exp'] = eid; v['experiment_id'] = f'{eid}#{tag}'
            v['sample_id'] = [f'{eid}#{tag}_{i}' for i in range(len(v))]
            parts.append(v)
    return pd.concat(parts, ignore_index=True)


class WXGB(exp_img.XGBBalanced):
    """balanced 클래스 가중치 × 행별 추가 가중치."""
    def fit(self, X, y, w_extra):
        y = np.asarray(y); self.classes_ = np.unique(y)
        cnt = np.array([(y == c).sum() for c in self.classes_], float)
        w = (len(y) / (len(self.classes_) * cnt))[np.searchsorted(self.classes_, y)] * w_extra
        self.m.fit(X.to_numpy(np.float32), np.searchsorted(self.classes_, y), sample_weight=w)
        return self


def gbm(seed):
    m = WXGB(seed)
    if os.environ.get('GBM_BACKEND') == 'xgb_cpu':
        m.m.set_params(device='cpu', n_jobs=int(os.environ.get('GBM_THREADS', '8')))
    return m


CFGS = {
    'S_oc_aug':     dict(oc2=False, jitter=False, w=1.0),
    'S_oc_aug_w03': dict(oc2=False, jitter=False, w=0.3),
    'S_oc2_w03':    dict(oc2=True, jitter=False, w=0.3),
    'S_oc2_j_w03':  dict(oc2=True, jitter=True, w=0.3),
    'S_oc2_j_w1':   dict(oc2=True, jitter=True, w=1.0),
}


COMBO = os.environ.get('COMBO') == '1'
if os.environ.get('CFG_LIST'):
    CFGS = {k: CFGS[k] for k in os.environ['CFG_LIST'].split(',')}


def feat_combo(fr):
    """셀 제품(bat_class×bat_cc, 6종) × 과충전(tr_method) 조합 원-핫 + 코드."""
    prod = fr.bat_class.astype(int).astype(str) + '_' + fr.bat_cc.astype(float).map('{:g}'.format)
    combo = prod + '_tm' + fr.tr_method.astype(int).astype(str)
    PRODS = ['1_4.8', '1_6', '2_39', '2_73', '3_28', '3_280']
    cols = {f'cb_prod_{q}': (prod == q).to_numpy(np.float32) for q in PRODS}
    for q in PRODS:
        for tm in (1, 2):
            cols[f'cb_{q}_tm{tm}'] = (combo == f'{q}_tm{tm}').to_numpy(np.float32)
    cols['cb_prod_code'] = prod.map({q: i for i, q in enumerate(PRODS)}).to_numpy(np.float32)
    cols['cb_combo_code'] = (cols['cb_prod_code'] * 2 + (fr.tr_method.to_numpy() == 2)).astype(np.float32)
    return pd.DataFrame(cols, index=fr.sample_id.values)


def fit_ckpt(path, seed, X, y, w):
    """학습한 모델은 항상 저장. 같은 경로에 체크포인트가 있으면 다시 학습하지 않고 불러옴(재개)."""
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return joblib.load(path)
    m = gbm(seed).fit(X, y, w)
    tmp = path.with_name(path.name + '.tmp')
    joblib.dump(m, tmp); tmp.replace(path)
    return m


def cnn(path, ids):
    c = np.load(path, allow_pickle=True)
    return pd.DataFrame(c['probs'], index=c['sample_id'].astype(str)).loc[ids].to_numpy()


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    for f in (tr, va, te):
        f['sample_id'] = f.sample_id.astype(str)
    tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)
    aug = {j: (make_aug2(tr, 0, j), make_aug2(va, 1, j)) for j in (False, True)}
    log('aug rows', {j: (len(a), len(b)) for j, (a, b) in aug.items()})

    def feats(f):
        parts = [sensor_feats(f), feat_oc2(f)] + ([feat_combo(f)] if COMBO else [])
        return pd.concat(parts, axis=1)
    S = {nm: feats(f) for nm, f in (('tr', tr), ('tr_cap', tr_cap), ('va', va), ('va_cap', va_cap), ('te', te))}
    SA = {(j, k): feats(a) for j, (atr, ava) in aug.items() for k, a in (('tr', atr), ('va', ava))}
    oc2_cols = [c for c in S['tr'].columns if c.startswith('oc2_')]
    base_cols = [c for c in S['tr'].columns if not c.startswith('oc2_')]
    log(f'feats base {len(base_cols)} + OC2 {len(oc2_cols)}')
    ytr = tr.label.astype(int).to_numpy()

    def design(Sf, Sa, ids, af, cfg):
        cols = base_cols + (oc2_cols if cfg['oc2'] else [])
        X = pd.concat([Sf.loc[ids, cols], Sa.loc[af.sample_id.values, cols]])
        w = np.r_[np.ones(len(ids)), np.full(len(af), cfg['w'])]
        return X, w, cols

    FINAL_ONLY = os.environ.get('FINAL_ONLY')   # CV·Val 생략, 지정 설정만 최종 학습
    # CV
    c_tr = cnn('ft_convnext_tiny.npz', tr_cap.sample_id.values)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    res = {}
    res_path = OUT / 'results.json'
    for nm, cfg in ({} if FINAL_ONLY else CFGS).items():
        P = np.zeros((len(tr_cap), 6))
        atr = aug[cfg['jitter']][0]
        for s in CV_SEEDS:
            for k, (a, b) in enumerate(splits):
                ea = set(tr.experiment_id.iloc[a]); af = atr[atr.src_exp.isin(ea)]
                X, w, cols = design(S['tr'], SA[(cfg['jitter'], 'tr')], tr.sample_id.values[a], af, cfg)
                y = np.r_[ytr[a], af.label.astype(int).to_numpy()]
                m = ~tr_cap.experiment_id.isin(ea).to_numpy()
                P[m] += fit_ckpt(OUT / 'ckpt' / nm / f'cv_seed{s}_fold{k}.joblib', s, X, y, w).predict_proba(S['tr_cap'].loc[tr_cap.sample_id.values[m], cols]) / len(CV_SEEDS)
        cv, cv_oc, pc = score(tr_cap, (1 - W_IMG) * P + W_IMG * c_tr)
        res[nm] = dict(cv=cv, cv_oc=cv_oc, cv_per_class=pc)
        np.save(OUT / f'oof_{nm}.npy', P)
        log(f'CV {nm:13s} +CNX0.1: all {cv:.4f}  OC {cv_oc:.4f}  {pc}')
        res_path.write_text(json.dumps(res, indent=1))

    # Val
    c_va = cnn('ft_convnext_tiny.npz', va_cap.sample_id.values)
    for nm, cfg in ({} if FINAL_ONLY else CFGS).items():
        atr = aug[cfg['jitter']][0]
        X, w, cols = design(S['tr'], SA[(cfg['jitter'], 'tr')], tr.sample_id.values, atr, cfg)
        y = np.r_[ytr, atr.label.astype(int).to_numpy()]
        pS = sum(fit_ckpt(OUT / 'ckpt' / nm / f'val_seed{s}.joblib', s, X, y, w).predict_proba(S['va_cap'].loc[va_cap.sample_id.values, cols]) for s in CV_SEEDS) / len(CV_SEEDS)
        v, v_oc, pc = score(va_cap, (1 - W_IMG) * pS + W_IMG * c_va)
        res[nm].update(val=v, val_oc=v_oc, val_per_class=pc)
        log(f'Val {nm:13s} +CNX0.1: all {v:.4f}  OC {v_oc:.4f}  {pc}')
        res_path.write_text(json.dumps(res, indent=1))

    if FINAL_ONLY:
        best = FINAL_ONLY
    else:
        top = max(r['cv'] for r in res.values())
        best = max([n for n in CFGS if res[n]['cv'] >= top - 0.0005], key=lambda n: res[n]['cv_oc'])
        res['_selected'] = best
        res_path.write_text(json.dumps(res, indent=1))
    log(f'selected: {best}')

    # 최종
    import joblib
    trva = pd.concat([tr, va], ignore_index=True)
    ytv = trva.label.astype(int).to_numpy()
    Stv = pd.concat([S['tr'], S['va']])
    c_te = cnn('ft_convnext_tiny_trval.npz', te.sample_id.values)
    ss = pd.read_csv(D / 'sample_submission.csv', dtype={'sample_id': str})
    for nm in ([best] if FINAL_ONLY else dict.fromkeys([best, 'S_oc_aug'])):
        cfg = CFGS[nm]
        atv = pd.concat(aug[cfg['jitter']], ignore_index=True)
        Satv = pd.concat([SA[(cfg['jitter'], 'tr')], SA[(cfg['jitter'], 'va')]])
        X, w, cols = design(Stv, Satv, trva.sample_id.values, atv, cfg)
        y = np.r_[ytv, atv.label.astype(int).to_numpy()]
        mdir = OUT / f'models_{nm}'; mdir.mkdir(exist_ok=True)
        joblib.dump(dict(cols=cols, cfg=cfg, img_w=W_IMG, img_model='ft_convnext_tiny_trval_models', tau=0.7), mdir / 'meta.joblib')
        pS = np.zeros((len(te), 6))
        for s in FINAL_SEEDS:
            m = fit_ckpt(mdir / f'xgb_seed{s}.joblib', s, X, y, w)
            pS += m.predict_proba(S['te'].loc[te.sample_id.values, cols]) / len(FINAL_SEEDS)
        np.save(OUT / f'test_probs_{nm}.npy', pS)
        lab = F.postprocess(te, (1 - W_IMG) * pS + W_IMG * c_te, 0.7)
        sub = ss[['sample_id']].merge(pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab}), on='sample_id', how='left')
        assert sub.label.notna().all() and len(sub) == len(ss)
        p = OUT / f'submission_{nm}_convnext0.1.csv'
        sub.to_csv(p, index=False)
        if nm == best and not FINAL_ONLY:
            sub.to_csv(OUT / 'submission_selected.csv', index=False)
        log(f'wrote {p.name} dist {sub.label.value_counts().sort_index().to_dict()}')
    log('ALL DONE')


if __name__ == '__main__':
    main()
