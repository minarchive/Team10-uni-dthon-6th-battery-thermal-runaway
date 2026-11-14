"""과충전(tr_method=2) 분포 차이 대응 + 이미지 가중치 0.1 고정 결합.

배경 (Train/Val 라벨·메타와 Test의 공개 구조만 사용, Test 값으로 학습·fit 없음)
- tr_method=2 = 과충전: 전압 3.7→~7V 상승, L1(충전 구간)이 길고 실험이 김.
- Test의 긴 실험 2개(5705s, 6240s)가 모두 과충전이고 500행 캡(10/15초 간격). Train+Val 최장은 약 6060s.

모델
- 센서 모델 S: v3 + 궤적 A + 과충전 피처(OC). 옵션: 과충전 증강(L1 시간 늘리기 ×1.2~1.6 + 500행 캡, 원본 캡)
- 이미지 모델 I: DINOv2-B 코사인 9 + PCA16 윈도우 피처만 사용
- 최종 확률 = (1-w)·S + w·I,  w = IMG_W (기본 0.1)  → 확신 cummax τ=0.7

평가: Train GroupKFold(5, seed 0) × GBM 시드 3, 평가 fold 500행 캡. 전체 / 과충전 실험만 Macro F1.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import final_mm as M
from final_mm import F, load, gbm, fit_pca, pca_feats, EMB, img_features
from exp_ctx import feat_traj, exp_groups, since_first

D = M.D
LAB = F.LABELS
TAG = os.environ.get('TAG', 'gpu')
IMG_W = float(os.environ.get('IMG_W', '0.1'))
CV_SEEDS = [int(s) for s in os.environ.get('CV_SEEDS', '0,1,2').split(',')]
FINAL_SEEDS = [int(s) for s in os.environ.get('FINAL_SEEDS', '0,1,2,3,4').split(',')]
N_STRETCH = int(os.environ.get('N_STRETCH', '2'))
OUT = M.HERE / f'final_oc_{TAG}'
OUT.mkdir(exist_ok=True)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


# ---------------------------------------------------------------------------
# 과충전 피처 (인과: 실험 내 현재 시점 이하만)
# ---------------------------------------------------------------------------
V_TH = (0.3, 0.6, 1.0, 1.5, 2.0, 3.0)


def feat_oc(fr):
    t, groups = exp_groups(fr)
    v = pd.to_numeric(fr.meas_v).to_numpy(float)
    t1 = pd.to_numeric(fr.meas_t1).to_numpy(float)
    n = len(fr)
    rise, rise_cm, rate, drop_frac, t1_since_vpk, vpk_rise = (np.full(n, np.nan) for _ in range(6))
    for idx in groups:
        vv, tt, tt1 = v[idx], t[idx], t1[idx]
        v0 = vv[np.isfinite(vv)][0] if np.isfinite(vv).any() else np.nan
        r = vv - v0
        cm = np.fmax.accumulate(np.where(np.isfinite(r), r, -np.inf))
        cm = np.where(np.isfinite(cm), cm, np.nan)
        rise[idx], rise_cm[idx] = r, cm
        rate[idx] = cm / np.maximum(tt, 5.0)
        drop_frac[idx] = (cm - r) / np.maximum(cm, 0.05)
        pk = np.maximum.accumulate(np.where(np.isfinite(r) & (r >= cm), np.arange(len(idx)), 0))
        t1_since_vpk[idx] = tt1 - tt1[pk]
        vpk_rise[idx] = cm
    cols = {'oc_v_rise': rise, 'oc_v_rise_cummax': rise_cm, 'oc_v_rise_rate': rate,
            'oc_v_drop_frac': drop_frac, 'oc_t1_since_vpeak': t1_since_vpk}
    for th in V_TH:
        cols[f'oc_since_vrise{th}'] = since_first(t, groups, np.nan_to_num(rise, nan=-np.inf) >= th)
    return pd.DataFrame(cols, index=fr.sample_id.values).astype(np.float32)


def sensor_feats(fr):
    X = F.build_features(fr).set_axis(fr.sample_id.values)
    return pd.concat([X, feat_traj(fr, X).set_axis(fr.sample_id.values), feat_oc(fr)], axis=1)


def image_feats(fr, pca):
    c = img_features(fr, EMB['B'], None, 0, use_pca=False, use_cos=True).add_prefix('B_')
    return pd.concat([c, pca_feats(fr, pca, 'B')], axis=1).loc[fr.sample_id.values]


# ---------------------------------------------------------------------------
# 과충전 증강: L1 구간 시간 늘리기 + Test식 500행 캡
# ---------------------------------------------------------------------------
def cap_idx(n, m=F.MAX_ROWS):
    return np.arange(n) if n <= m else np.unique(np.round(np.linspace(0, n - 1, m)).astype(int))


def stretch_l1(e, s):
    """L1 구간 시간축을 s배로 늘림 (5초 격자, 직전 관측값 유지). L1 이후는 시간만 평행이동."""
    e = e.sort_values('relative_time_s')
    t = e.relative_time_s.to_numpy(float)
    l1 = e.label.to_numpy() == 1
    t_end = t[l1].max() if l1.any() else 0.0
    new_t = np.arange(0, t_end * s + 1e-9, 5.0)
    src = np.searchsorted(t, new_t / s, side='right') - 1
    pre = e.iloc[np.clip(src, 0, None)].copy()
    pre['relative_time_s'] = new_t
    post = e[t > t_end].copy()
    post['relative_time_s'] = post.relative_time_s + t_end * (s - 1)
    return pd.concat([pre, post])


def make_aug(fr, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for eid, e in fr[fr.tr_method == 2].groupby('experiment_id'):
        variants = [(0, 1.0)] + [(k + 1, rng.uniform(1.2, 1.6)) for k in range(N_STRETCH)]
        for k, s in variants:
            v = stretch_l1(e, s) if s != 1.0 else e.sort_values('relative_time_s')
            v = v.iloc[cap_idx(len(v))].copy()
            if k == 0 and len(e) <= F.MAX_ROWS:
                continue                                   # 원본과 같으면 건너뜀
            v['src_exp'] = eid
            v['experiment_id'] = f'{eid}#a{k}'
            v['sample_id'] = [f'{eid}#a{k}_{i}' for i in range(len(v))]
            parts.append(v)
    return pd.concat(parts, ignore_index=True)


def macro(y, p):
    return float(f1_score(y, p, labels=LAB, average='macro'))


def score(fr, P):
    lab = F.postprocess(fr, P, 0.7)
    y = fr.label.astype(int).to_numpy()
    oc = (fr.tr_method == 2).to_numpy()
    return macro(y, lab), macro(y[oc], lab[oc]), f1_score(y, lab, labels=LAB, average=None).round(4).tolist()


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    for f in (tr, va, te):
        f['sample_id'] = f.sample_id.astype(str)
    tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)
    aug_tr, aug_va = make_aug(tr, 0), make_aug(va, 1)
    log(f'aug rows: train {len(aug_tr)} ({aug_tr.experiment_id.nunique()} variants), val {len(aug_va)}')
    S = {nm: sensor_feats(f) for nm, f in (('tr', tr), ('tr_cap', tr_cap), ('va', va), ('va_cap', va_cap),
                                           ('te', te), ('aug_tr', aug_tr), ('aug_va', aug_va))}
    log(f'sensor feats {S["tr"].shape[1]} (OC {sum(c.startswith("oc_") for c in S["tr"].columns)})')
    ytr = tr.label.astype(int).to_numpy()
    no_oc = [c for c in S['tr'].columns if not c.startswith('oc_')]

    sensor_cfgs = {'S_v3A': dict(cols=no_oc, aug=False), 'S_oc': dict(cols=None, aug=False),
                   'S_oc_aug': dict(cols=None, aug=True)}

    def sens_train(Sf, Sa, fr_ids, aug_fr, cfg):
        X = Sf.loc[fr_ids]
        y = None
        if cfg['aug'] and len(aug_fr):
            X = pd.concat([X, Sa.loc[aug_fr.sample_id.values]])
        cols = cfg['cols'] or list(X.columns)
        return X[cols], cols

    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    res = {}
    oof = {k: np.zeros((len(tr_cap), 6)) for k in list(sensor_cfgs) + ['I']}
    for s in CV_SEEDS:
        for a, b in splits:
            fa = tr.iloc[a]; ea = set(fa.experiment_id)
            fb_mask = ~tr_cap.experiment_id.isin(ea).to_numpy()
            fb = tr_cap[fb_mask]
            # 이미지 모델
            pca = fit_pca(fa, 'B')
            Ia, Ib = image_feats(fa, pca), image_feats(fb, pca)
            oof['I'][fb_mask] += gbm(s).fit(Ia, ytr[a]).predict_proba(Ib) / len(CV_SEEDS)
            # 센서 모델들
            af = aug_tr[aug_tr.src_exp.isin(ea)]
            for nm, cfg in sensor_cfgs.items():
                Xa, cols = sens_train(S['tr'], S['aug_tr'], fa.sample_id.values, af, cfg)
                ya = np.r_[ytr[a], af.label.astype(int).to_numpy()] if cfg['aug'] else ytr[a]
                oof[nm][fb_mask] += gbm(s).fit(Xa, ya).predict_proba(S['tr_cap'].loc[fb.sample_id.values, cols]) / len(CV_SEEDS)
        log(f'CV seed {s} done')
    for nm in sensor_cfgs:
        for w in (0.0, IMG_W):
            P = (1 - w) * oof[nm] + w * oof['I']
            cv, cv_oc, pc = score(tr_cap, P)
            res[f'{nm}|w{w}'] = dict(cv=cv, cv_oc=cv_oc, cv_per_class=pc)
            log(f'CV {nm:9s} img_w {w:.1f}: all {cv:.4f}  overcharge {cv_oc:.4f}  {pc}')
    res['I_only_cv'] = score(tr_cap, oof['I'])[:2]
    log(f'image-only CV {res["I_only_cv"]}')
    np.savez(OUT / 'oof.npz', **oof)

    # Val 확인 (Train 전체 학습)
    pca = fit_pca(tr, 'B')
    pI = sum(gbm(s).fit(image_feats(tr, pca), ytr).predict_proba(image_feats(va_cap, pca)) for s in CV_SEEDS) / len(CV_SEEDS)
    for nm, cfg in sensor_cfgs.items():
        Xa, cols = sens_train(S['tr'], S['aug_tr'], tr.sample_id.values, aug_tr, cfg)
        ya = np.r_[ytr, aug_tr.label.astype(int).to_numpy()] if cfg['aug'] else ytr
        pS = sum(gbm(s).fit(Xa, ya).predict_proba(S['va_cap'].loc[va_cap.sample_id.values, cols]) for s in CV_SEEDS) / len(CV_SEEDS)
        for w in (0.0, IMG_W):
            v, v_oc, pc = score(va_cap, (1 - w) * pS + w * pI)
            res[f'{nm}|w{w}'].update(val=v, val_oc=v_oc, val_per_class=pc)
            log(f'Val {nm:9s} img_w {w:.1f}: all {v:.4f}  overcharge {v_oc:.4f}  {pc}')
    (OUT / 'results.json').write_text(json.dumps(res, indent=1))

    # 선택: 이미지 w=IMG_W 고정, 센서 설정은 CV 전체 점수로
    best = max(sensor_cfgs, key=lambda n: res[f'{n}|w{IMG_W}']['cv'])
    log(f'selected sensor model by CV: {best} (+ image w={IMG_W})')

    # 최종: Train+Val 학습 → Test
    trva = pd.concat([tr, va], ignore_index=True)
    aug_tv = pd.concat([aug_tr, aug_va], ignore_index=True)
    Stv, Saug = pd.concat([S['tr'], S['va']]), pd.concat([S['aug_tr'], S['aug_va']])
    ytv = trva.label.astype(int).to_numpy()
    import joblib
    mdir = OUT / 'models'; mdir.mkdir(exist_ok=True)
    pca = fit_pca(trva, 'B')
    It, Ite = image_feats(trva, pca), image_feats(te, pca)
    joblib.dump(dict(pca=pca, img_cols=list(It.columns), img_w=IMG_W, tau=0.7, sensor_cfg=best), mdir / 'meta.joblib')
    out = {}
    for nm in dict.fromkeys([best, 'S_v3A']):
        cfg = sensor_cfgs[nm]
        Xa, cols = sens_train(Stv, Saug, trva.sample_id.values, aug_tv, cfg)
        ya = np.r_[ytv, aug_tv.label.astype(int).to_numpy()] if cfg['aug'] else ytv
        pS = np.zeros((len(te), 6))
        for s in FINAL_SEEDS:
            m = gbm(s).fit(Xa, ya); joblib.dump(m, mdir / f'{nm}_seed{s}.joblib')
            pS += m.predict_proba(S['te'].loc[te.sample_id.values, cols]) / len(FINAL_SEEDS)
        joblib.dump(cols, mdir / f'{nm}_cols.joblib')
        out[nm] = pS
        log(f'final sensor {nm} done')
    pI = np.zeros((len(te), 6))
    for s in FINAL_SEEDS:
        m = gbm(s).fit(It, ytv); joblib.dump(m, mdir / f'I_seed{s}.joblib')
        pI += m.predict_proba(Ite) / len(FINAL_SEEDS)
    log('final image model done')
    ss = pd.read_csv(D / 'sample_submission.csv', dtype={'sample_id': str})
    for nm, pS in out.items():
        P = (1 - IMG_W) * pS + IMG_W * pI
        lab = F.postprocess(te, P, 0.7)
        sub = ss[['sample_id']].merge(pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab}), on='sample_id', how='left')
        assert sub.label.notna().all() and len(sub) == len(ss)
        path = OUT / f'submission_{nm}_img{IMG_W}.csv'
        sub.to_csv(path, index=False)
        np.save(OUT / f'test_probs_{nm}.npy', pS)
        log(f'wrote {path.name}  dist {sub.label.value_counts().sort_index().to_dict()}')
        if nm == best:
            sub.to_csv(OUT / 'submission_selected.csv', index=False)
    np.save(OUT / 'test_probs_I.npy', pI)
    log('ALL DONE')


if __name__ == '__main__':
    main()
