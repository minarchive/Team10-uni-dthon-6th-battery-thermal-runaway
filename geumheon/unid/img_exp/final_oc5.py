"""oc4 R_J6wide 확장: (1) 트리 키우기 (2) Test 에서 분포가 달라지는 피처의 관련 피처 추가.

기준: final_oc4 R_J6wide (45°C 규칙 + 2~6단계 모델 + 시계 재설정, 과충전 증강 J6wide w 0.3, ConvNeXt 0.1)
       CV 0.9054 OC 0.8159 | Val 0.9105 OC 0.8524

트리 (config 이름 접미사)
  (없음): XGB lossguide 잎 31, 트리 500, lr 0.03 (oc4 와 동일)
  _B63  : 잎 63, 트리 1500, lr 0.02
  _B127 : 잎 127, 트리 2500, lr 0.015

OOD 관련 피처 (OOD=1 일 때 추가). ood_oc4.py 로 찾은 Test 분포 이동 피처(KS·범위 이탈 상위)의 원천 신호별:
  주변온도 atmp_t  : 튀는 값 [10,60] 클립 → 300초 이동 중앙값, 시작 30초 대비 변화, 각 온도 신호 − 주변온도
  *_cummin         : 온도 신호 − 누적최소 (시작 온도 차이 제거)
  가스 7종         : 시작 30초 평균 대비, 누적최대 − 시작값, 45°C 돌파 시점 대비, log1p
  oc_since*/oc2_since* : 45°C 돌파 후 경과 시간 대비 비율
  전압             : 누적최대 / 시작 30초 전압
모두 같은 실험의 현재·과거 행만 사용 (인과).

  TAG=oc5_base OC5_CFGS=R_J6wide_B63 python final_oc5.py
  TAG=oc5_ood OOD=1 OC5_CFGS=R_J6wide,R_J6wide_B63 python final_oc5.py
  OOD=1 python final_oc5.py ks       # 추가 피처의 Train+Val vs Test KS 확인
"""
import os
import sys

import numpy as np
import pandas as pd

import final_oc as O
import final_oc2 as Q2
import final_oc4 as Q4

OOD = os.environ.get('OOD') == '1'
TREES = {
    '':      dict(max_leaves=31, n_estimators=500, learning_rate=0.03),
    '_B63':  dict(max_leaves=63, n_estimators=1500, learning_rate=0.02),
    '_B127': dict(max_leaves=127, n_estimators=2500, learning_rate=0.015),
}
CFGS = {f'R_J6wide{k}': dict(aug='wide', w=0.3, reset=True, tree=k) for k in TREES}
if os.environ.get('OC5_CFGS'):
    CFGS = {k: CFGS[k] for k in os.environ['OC5_CFGS'].split(',')}

TEMPS = ['meas_t1', 'meas_t2', 'tmax', 'indu_t', 'thermal_t']
GASES = ['meas_no2', 'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hf', 'meas_no']


def ood_feats(fr, X):
    d = pd.DataFrame({'sid': fr.sample_id.astype(str).values, 'e': fr.experiment_id.astype(str).values,
                      't': fr.relative_time_s.astype(float).values})
    for c in ['meas_t1', 'meas_t2', 'indu_t', 'thermal_t', 'atmp_t', 'meas_v'] + GASES:
        d[c] = pd.to_numeric(fr[c], errors='coerce').values
    d['tmax'] = d[['meas_t1', 'meas_t2']].max(1)
    d = d.sort_values(['e', 't']).reset_index(drop=True)
    g = d.groupby('e', sort=False)
    d['_ts'] = pd.to_datetime(d.t, unit='s')

    def first30(s):
        return s.where(d.t <= 30).groupby(d.e).transform(lambda v: v.expanding().mean()).groupby(d.e).ffill()

    F = {}
    d['atmp_c'] = d.atmp_t.where(d.atmp_t.between(10, 60))
    amb = d.groupby('e', sort=False).rolling('300s', on='_ts').atmp_c.median().reset_index(drop=True)
    amb = amb.groupby(d.e).ffill()
    F['ood_amb_med300'] = amb
    F['ood_amb_rel0'] = amb - first30(d.atmp_c)
    for s in TEMPS:
        F[f'ood_{s}_over_amb'] = d[s] - amb
        F[f'ood_{s}_over_cmin'] = d[s] - g[s].cummin()
    hit = g.meas_t1.cummax() >= 45
    first_hit = hit & ~hit.groupby(d.e).shift(1, fill_value=False)
    for s in GASES:
        x = d[s]; b = first30(x)
        F[f'ood_{s}_rel0'] = x - b
        F[f'ood_{s}_cmax_rel0'] = g[s].cummax() - b
        F[f'ood_{s}_rel45'] = x - x.where(first_hit).groupby(d.e).ffill()
        F[f'ood_{s}_log'] = np.log1p(x.clip(lower=0))
    v0 = first30(d.meas_v)
    F['ood_v_cmax_rel0'] = g.meas_v.cummax() / v0.where(v0.abs() > 1e-3)
    out = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan).astype('float32')
    out.index = d.sid.values
    out = out.loc[fr.sample_id.astype(str).values]
    since = [c for c in X.columns if c.startswith(('oc_since', 'oc2_since')) and c != 'oc2_since45_over_t45']
    den = X['t_since45'].to_numpy(float) + 30.0
    R = pd.DataFrame({f'ood_{c}_frac45': (X[c].to_numpy(float) / den).astype('float32') for c in since}, index=X.index)
    return pd.concat([out, R], axis=1)


_feats_r = Q4.feats_r


def feats_r(fr):
    X, H = _feats_r(fr)
    if OOD:
        X = pd.concat([X, ood_feats(fr, X)], axis=1)
    return X, H


def fit_ckpt(path, seed, X, y, w):
    """체크포인트 경로의 config 이름으로 트리 크기 결정 (ckpt/<cfg>/...)."""
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return joblib.load(path)
    m = Q2.gbm(seed)
    m.m.set_params(**TREES[CFGS[path.parent.name]['tree']])
    m.fit(X, y, w)
    tmp = path.with_name(path.name + '.tmp')
    joblib.dump(m, tmp); tmp.replace(path)
    return m


def ks():
    from scipy.stats import ks_2samp
    from final_oc import F, load, D
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    for f in (tr, va, te):
        f['sample_id'] = f.sample_id.astype(str)
    tv = pd.concat([F.cap_rows(tr), F.cap_rows(va)], ignore_index=True)
    Xa, Ha = feats_r(tv); Xt, Ht = feats_r(te)
    Xa = Xa[Ha.to_numpy()]; Xt = Xt[Ht.to_numpy()]
    rows = []
    for c in [c for c in Xa.columns if c.startswith('ood_')]:
        a = Xa[c].to_numpy(float); t = Xt[c].to_numpy(float); a = a[np.isfinite(a)]; t = t[np.isfinite(t)]
        if len(a) < 10 or len(t) < 10:
            continue
        rows.append(dict(feat=c, ks=ks_2samp(a, t).statistic, frac_out=((t < a.min()) | (t > a.max())).mean(),
                         tv_med=np.median(a), te_med=np.median(t)))
    R = pd.DataFrame(rows).sort_values('ks')
    pd.set_option('display.width', 200)
    print(f'ood feats {len(R)} | KS median {R.ks.median():.3f}')
    print(R.round(3).to_string(index=False))


Q4.feats_r = feats_r
Q4.fit_ckpt = fit_ckpt
Q4.CFGS = CFGS
Q4.OUT = O.M.HERE / f'final_oc5_{os.environ.get("TAG", "gpu")}'
Q4.OUT.mkdir(exist_ok=True)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'ks':
        ks()
    else:
        Q4.main()
