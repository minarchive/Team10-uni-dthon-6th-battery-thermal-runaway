"""최종 멀티모달(센서 시계열 + 열화상 이미지) 모델: 선택(Train CV) → 확인(Val) → Train+Val 학습 → Test 제출.

규칙
- Test는 추론에만 사용 (학습·의사라벨·전처리 fit 모두 Train(+Val)만).
- 모든 피처는 같은 실험의 현재 시점 이하 프레임만 사용 (인과). 후처리는 확신 cummax τ=0.7 (온라인).
- 모델 선택은 Train GroupKFold(5) CV(500행 캡 + 후처리)로만. Val은 확인용.

피처 세트
  v3A      : v3 센서 시계열(735) + 궤적 맥락 A
  v3A+cB   : + DINOv2-B 코사인 거리 피처 (첫 프레임/직전/30·60·120초 전)
  v3A+cBS  : + DINOv2-S 코사인 피처
  v3A+cB+zB: + DINOv2-B PCA16 성분의 윈도우 평균·변화 (PCA는 학습 행으로만 fit)

환경변수: GBM_BACKEND=xgb_cuda|xgb_cpu, GBM_THREADS, TAG, CV_SEEDS, FINAL_SEEDS
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
import exp_img  # noqa: E402
from exp_img import F, load, img_features  # noqa: E402
from exp_ctx import feat_traj  # noqa: E402

D = Path('/home/nxcolelxu/unid/data/battery_data_light')
TAG = os.environ.get('TAG', 'gpu')
BACKEND = os.environ.get('GBM_BACKEND', 'xgb_cuda')
CV_SEEDS = [int(s) for s in os.environ.get('CV_SEEDS', '0,1,2').split(',')]
FINAL_SEEDS = [int(s) for s in os.environ.get('FINAL_SEEDS', '0,1,2,3,4').split(',')]
LAB = F.LABELS
OUT = HERE / f'final_mm_{TAG}'
OUT.mkdir(exist_ok=True)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def gbm(seed):
    m = exp_img.XGBBalanced(seed)
    if BACKEND == 'xgb_cpu':
        m.m.set_params(device='cpu', n_jobs=int(os.environ.get('GBM_THREADS', '8')))
    return m


def load_emb(path):
    d = np.load(path, allow_pickle=True)
    return pd.DataFrame(np.hstack([d['cls'], d['patch_mean']]).astype(np.float32), index=d['sample_id'].astype(str))


EMB = {'B': load_emb('emb_dinov2b.npz'), 'S': load_emb('emb_dinov2s.npz')}
CONFIGS = {
    'v3A': dict(cos=[], pca=None),
    'v3A+cB': dict(cos=['B'], pca=None),
    'v3A+cBS': dict(cos=['B', 'S'], pca=None),
    'v3A+cB+zB': dict(cos=['B'], pca='B'),
    # Test 분포 맞추기: Test는 실험당 최대 500행 균등 다운샘플 → 학습 데이터도 같은 조건으로 생성
    'v3A+cB_capaug': dict(cos=['B'], pca=None, train='aug'),
    'v3A+cB_caponly': dict(cos=['B'], pca=None, train='cap'),
}
PCA_K = 16


def base_feats(fr):
    """설정과 무관한 피처(센서 v3 + A + 코사인)를 한 번만 계산."""
    X = F.build_features(fr)
    X = X.set_axis(fr.sample_id.values)
    parts = {'v3A': pd.concat([X, feat_traj(fr, X).set_axis(fr.sample_id.values)], axis=1)}
    for e in ('B', 'S'):
        c = img_features(fr, EMB[e], None, 0, use_pca=False, use_cos=True)
        parts[f'c{e}'] = c.add_prefix(f'{e}_')
    return parts


def pca_feats(fr, pca, e):
    c = img_features(fr, EMB[e], pca, PCA_K, use_pca=True, use_cos=False)
    return c.add_prefix(f'{e}_')


def design(parts, cfg, fr=None, pca=None):
    X = [parts['v3A']] + [parts[f'c{e}'] for e in cfg['cos']]
    if cfg['pca']:
        X.append(pca_feats(fr, pca, cfg['pca']))
    return pd.concat(X, axis=1).loc[fr.sample_id.values]


def train_X(Pf, Pc, ff, fc, cfg, pca, y_col='label'):
    """학습 행렬. train='full'(원본) | 'cap'(Test식 500행 캡 프레임) | 'aug'(원본 + 캡이 실제로 걸린 실험의 캡 프레임)"""
    mode = cfg.get('train', 'full')
    sub = lambda Pp, fr: {k: v.loc[fr.sample_id.values] for k, v in Pp.items()}
    if mode == 'cap':
        return design(sub(Pc, fc), cfg, fc, pca), fc[y_col].astype(int).to_numpy()
    Xf = design(sub(Pf, ff), cfg, ff, pca); yf = ff[y_col].astype(int).to_numpy()
    if mode == 'full':
        return Xf, yf
    n = ff.groupby('experiment_id').size()
    fc2 = fc[fc.experiment_id.isin(n[n > F.MAX_ROWS].index)]
    Xc = design(sub(Pc, fc2), cfg, fc2, pca)
    return pd.concat([Xf.reset_index(drop=True), Xc.reset_index(drop=True)]), np.r_[yf, fc2[y_col].astype(int).to_numpy()]


def fit_pca(fr, e):
    return PCA(PCA_K, random_state=0).fit(EMB[e].loc[fr.sample_id.values].to_numpy(np.float32))


def macro(y, p):
    return float(f1_score(y, p, labels=LAB, average='macro'))


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    for fr in (tr, va, te):
        fr['sample_id'] = fr.sample_id.astype(str)
    log(f'backend {BACKEND}  rows tr {len(tr)} va {len(va)} te {len(te)}')
    # 평가 프레임은 Test 조건(500행 캡)으로 피처를 다시 계산
    tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)
    P = {nm: base_feats(fr) for nm, fr in (('tr', tr), ('tr_cap', tr_cap), ('va', va), ('va_cap', va_cap), ('te', te))}
    log('features:', {k: v.shape[1] for k, v in P['tr'].items()})
    ytr = tr.label.astype(int).to_numpy()
    yva_cap = va_cap.label.astype(int).to_numpy()
    cap_ids = set(tr_cap.sample_id)

    res_path = OUT / 'results.json'
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))

    ONLY_FINAL = os.environ.get('ONLY_FINAL')   # CV 생략, 지정 설정만 최종 학습 (CPU 대체 경로)
    # ---------------- 1) Train CV로 선택 ----------------
    for name, cfg in ({} if ONLY_FINAL else CONFIGS).items():
        if name in res and 'val' in res[name]:
            log(f'skip {name} (done)'); continue
        oof_avg = np.zeros((len(tr_cap), 6)); cvs = []
        for s in CV_SEEDS:
            oof = np.zeros((len(tr_cap), 6))
            for a, b in splits:
                fa = tr.iloc[a]
                ev_ids = set(tr.sample_id.iloc[b]) & cap_ids
                fb = tr_cap[tr_cap.sample_id.isin(ev_ids)]
                fa_cap = tr_cap[tr_cap.experiment_id.isin(set(fa.experiment_id))]
                pca = fit_pca(fa, cfg['pca']) if cfg['pca'] else None
                Xa, ya = train_X(P['tr'], P['tr_cap'], fa, fa_cap, cfg, pca)
                Xb = design({k: v.loc[fb.sample_id.values] for k, v in P['tr_cap'].items()}, cfg, fb, pca)
                oof[fb.index.to_numpy()] = gbm(s).fit(Xa, ya).predict_proba(Xb)
            post = F.postprocess(tr_cap, oof, 0.7)
            cvs.append(macro(tr_cap.label.astype(int).to_numpy(), post))
            oof_avg += oof / len(CV_SEEDS)
            log(f'{name} seed {s} CV post {cvs[-1]:.4f}')
        cv_ens = macro(tr_cap.label.astype(int).to_numpy(), F.postprocess(tr_cap, oof_avg, 0.7))
        # Val 확인: Train 전체로 학습
        pca = fit_pca(tr, cfg['pca']) if cfg['pca'] else None
        (Xtr, ytr_m), Xv = train_X(P['tr'], P['tr_cap'], tr, tr_cap, cfg, pca), design(P['va_cap'], cfg, va_cap, pca)
        pv = np.zeros((len(va_cap), 6))
        for s in CV_SEEDS:
            pv += gbm(s).fit(Xtr, ytr_m).predict_proba(Xv) / len(CV_SEEDS)
        vpost = F.postprocess(va_cap, pv, 0.7)
        res[name] = dict(n_feat=int(Xtr.shape[1]), cv_seeds=cvs, cv_mean=float(np.mean(cvs)), cv_std=float(np.std(cvs)),
                         cv_ens=cv_ens, val=macro(yva_cap, vpost), val_raw=macro(yva_cap, LAB[pv.argmax(1)]),
                         val_per_class=f1_score(yva_cap, vpost, labels=LAB, average=None).round(4).tolist())
        r = res[name]
        log(f'== {name:10s} feats {r["n_feat"]}  CV {r["cv_mean"]:.4f}±{r["cv_std"]:.4f} (ens {cv_ens:.4f})  '
            f'Val {r["val"]:.4f} (raw {r["val_raw"]:.4f}) {r["val_per_class"]}')
        np.savez(OUT / f'oof_{name}.npz', oof=oof_avg, val=pv)
        res_path.write_text(json.dumps(res, indent=1))

    # 선택: 이미지 포함 설정 중 CV 평균 1위 (Val은 보지 않음)
    img_cfgs = [n for n in CONFIGS if n != 'v3A']
    best = ONLY_FINAL or max(img_cfgs, key=lambda n: res[n]['cv_mean'])
    # 학습 데이터 구성(full/aug/cap) 선택도 CV로: 최고 피처 세트에 best 학습 모드 결합 여부는 위 후보 중 1위가 결정
    log(f'selected: {best}' + ('' if ONLY_FINAL else f'  (v3A reference CV {res["v3A"]["cv_mean"]:.4f})'))
    res['_selected'] = best
    res_path.write_text(json.dumps(res, indent=1))

    # ---------------- 2) Train+Val 학습 → Test ----------------
    trva = pd.concat([tr, va], ignore_index=True)
    trva_cap = pd.concat([tr_cap, va_cap], ignore_index=True)
    Ptv = {k: pd.concat([P['tr'][k], P['va'][k]]) for k in P['tr']}
    Ptv_cap = {k: pd.concat([P['tr_cap'][k], P['va_cap'][k]]) for k in P['tr']}
    for name in [best] + ([] if ONLY_FINAL else [n for n in CONFIGS if n != best]):
        sub_path = OUT / f'submission_{name.replace("+", "_")}.csv'
        if sub_path.exists():
            log(f'skip final {name}'); continue
        cfg = CONFIGS[name]
        pca = fit_pca(trva, cfg['pca']) if cfg['pca'] else None
        (Xtv, ytv), Xte = train_X(Ptv, Ptv_cap, trva, trva_cap, cfg, pca), design(P['te'], cfg, te, pca)
        pt = np.zeros((len(te), 6))
        mdir = OUT / f'models_{name.replace("+", "_")}'
        mdir.mkdir(exist_ok=True)
        import joblib
        joblib.dump(dict(columns=list(Xtv.columns), cfg=cfg, pca=pca, tau=0.7), mdir / 'meta.joblib')
        for s in FINAL_SEEDS:
            mp = mdir / f'xgb_seed{s}.joblib'
            if mp.exists():                       # 시드 단위 재개
                m = joblib.load(mp)
            else:
                m = gbm(s).fit(Xtv, ytv); joblib.dump(m, mp)
            pt += m.predict_proba(Xte) / len(FINAL_SEEDS)
            log(f'final {name} seed {s} done -> {mp.name}')
        lab = F.postprocess(te, pt, 0.7)
        sub = pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab.astype(int)})
        ss = pd.read_csv(D / 'sample_submission.csv', dtype={'sample_id': str})
        sub = ss[['sample_id']].merge(sub, on='sample_id', how='left')
        assert sub.label.notna().all() and len(sub) == len(ss)
        sub.to_csv(sub_path, index=False)
        np.save(OUT / f'test_probs_{name.replace("+", "_")}.npy', pt)
        log(f'wrote {sub_path}  label dist {sub.label.value_counts().sort_index().to_dict()}')
        if name == best:
            sub.to_csv(OUT / 'submission_selected.csv', index=False)
    log('ALL DONE')


if __name__ == '__main__':
    main()
