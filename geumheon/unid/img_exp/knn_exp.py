"""k-최근접 실험 모델: 얼마나 '국소적으로' 학습해야 좋은가 (Train+Val 97개 실험 leave-one-experiment-out).

빠진 실험 e 마다 나머지 96개로 학습해 e(500행 캡 = Test 조건)를 예측한다.
모델: SOTA 0.8812 LightGBM 설정·rich 피처 166개. e 가 과충전(tr_method 2)이면 C 피처(압력 절대값·v_peak_ratio 제외), 아니면 A.

변형
  global      : 96개 전부 (기준)
  local_k{k}  : e 와 가장 비슷한 k개 실험만으로 학습 (k = 1, 5, 15, 30)
  wk15        : 96개 전부, 가장 비슷한 15개 행에 가중치 5배

실험 유사도 (인과: 메타정보 + 시작 30초 = 45°C 규칙상 항상 1단계 구간)
  tr_method 가 다르면 제외. 거리 = 표준화한 [bat_class(×3), bat_cc, soc_status, 시작 30초 평균 t1·t2·p·v·thermal_t] 유클리드.

점수: 97개 OOF 모아 macro F1. 이미지 ConvNeXt-tiny(ft_convnext_tiny.npz, 해당 실험 미학습) α 0.1 결합 + 센서 단독,
      SOTA 인과 단조 Viterbi. 그룹: 전체 / 과충전 / tm1 / 시작 압력 높은 실험(p0>8) / Val 실험 19개

  python knn_exp.py run      # 학습·예측 (병렬, 체크포인트 = 실험별 npy)
  python knn_exp.py score
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

SOTA = Path('/home/nxcolelxu/Team10-uni-dthon-6th-battery-thermal-runaway/sota_0.8812')
sys.path.insert(0, str(SOTA))
import features as FT  # noqa: E402
import model as SM  # noqa: E402

D = Path('/home/nxcolelxu/unid/data/battery_data_light')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'knn_exp'
(OUT / 'pred').mkdir(parents=True, exist_ok=True)
CNX = HERE / 'ft_convnext_tiny.npz'
KS = (1, 5, 15, 30)
VARIANTS = ['global'] + [f'local_k{k}' for k in KS] + ['wk15']
N_PAR = int(os.environ.get('KNN_PAR', '4'))
THREADS = int(os.environ.get('KNN_THREADS', '4'))
LAB = np.arange(1, 7)
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


def data():
    tr = pd.read_csv(D / 'train' / 'train.csv'); va = pd.read_csv(D / 'validation' / 'validation.csv')
    tr['_src'] = 'train'; va['_src'] = 'validation'
    df = pd.concat([tr, va], ignore_index=True)
    return df, cap_rows(df)


def feats(frame, name):
    p = OUT / f'feat_{name}.pkl'
    if p.exists():
        return pd.read_pickle(p)
    X = FT.build(frame, 'rich'); X.to_pickle(p); log(f'features {name} {X.shape}')
    return X


def exp_table(df):
    """실험별 메타정보 + 시작 30초 평균"""
    g = df[df.relative_time_s <= 30].groupby('experiment_id')
    E = df.groupby('experiment_id')[['tr_method', 'bat_class', 'bat_cc', 'soc_status', '_src']].first()
    for c in ['meas_t1', 'meas_t2', 'meas_p', 'meas_v', 'thermal_t']:
        E[f'{c}_0'] = g[c].apply(lambda s: pd.to_numeric(s, errors='coerce').mean())
    return E


def neighbors(E):
    cols = ['bat_cc', 'soc_status', 'meas_t1_0', 'meas_t2_0', 'meas_p_0', 'meas_v_0', 'thermal_t_0']
    Z = (E[cols] - E[cols].mean()) / E[cols].std()
    Z = Z.fillna(0).to_numpy()
    bc = E.bat_class.to_numpy(); tm = E.tr_method.to_numpy(); ids = E.index.to_numpy()
    nb = {}
    for i, e in enumerate(ids):
        d = np.sqrt(((Z - Z[i]) ** 2).sum(1) + 9.0 * (bc != bc[i]))
        d[tm != tm[i]] = np.inf; d[i] = np.inf
        o = np.argsort(d); o = o[np.isfinite(d[o])]
        nb[e] = list(ids[o])
    return nb


def fit_predict(e, df, cap, Xtr, Xev, cols_a, cols_c, nb, tm):
    import lightgbm as lgb
    todo = [v for v in VARIANTS if not (OUT / 'pred' / f'{v}__{e}.npy').exists()]
    if not todo:
        return e
    cols = cols_c if tm == 2 else cols_a
    ev = cap.experiment_id.values == e
    Xe = Xev.loc[cap.sample_id.values[ev], cols]
    exp = df.experiment_id.values; y = df.label.values.astype(int)
    params = dict(SM.LGB_PARAMS, n_jobs=THREADS)

    def run(mask, w=None):
        m = lgb.LGBMClassifier(**params).fit(Xtr.loc[mask, cols], y[mask], sample_weight=w)
        P = np.zeros((ev.sum(), 6)); P[:, np.searchsorted(LAB, m.classes_)] = m.predict_proba(Xe)
        return P
    others = exp != e
    for v in todo:
        if v == 'global':
            P = run(others)
        elif v.startswith('local_k'):
            k = int(v[7:]); P = run(np.isin(exp, nb[e][:k]))
        elif v == 'wk15':
            w = np.where(np.isin(exp[others], nb[e][:15]), 5.0, 1.0); P = run(others, w)
        np.save(OUT / 'pred' / f'{v}__{e}.npy', P)
    return e


def run():
    df, cap = data()
    Xtr = feats(df, 'trval'); Xtr.index = np.arange(len(df))     # 행 위치 인덱스
    Xev = feats(cap, 'trval_cap')
    cols_a = list(Xtr.columns); cols_c = SM.cols_without_abs_pressure(cols_a)
    E = exp_table(df); nb = neighbors(E)
    json.dump({e: v[:30] for e, v in nb.items()}, open(OUT / 'neighbors.json', 'w'), indent=0)
    log(f'exps {len(E)} | A {len(cols_a)} C {len(cols_c)} | par {N_PAR} x threads {THREADS}')
    ids = list(E.index)
    done = 0
    for e in Parallel(n_jobs=N_PAR, return_as='generator')(
            delayed(fit_predict)(e, df, cap, Xtr, Xev, cols_a, cols_c, nb, E.tr_method[e]) for e in ids):
        done += 1
        if done % 10 == 0 or done == len(ids):
            log(f'{done}/{len(ids)} exps done')


def score():
    df, cap = data()
    E = exp_table(df)
    c = np.load(CNX, allow_pickle=True)
    V = pd.DataFrame(c['probs'], index=c['sample_id'].astype(str)).loc[cap.sample_id.astype(str).values].to_numpy()
    y = cap.label.values.astype(int)
    p0 = cap.experiment_id.map(E.meas_p_0).values
    G = dict(all=np.ones(len(y), bool), OC=cap.tr_method.values == 2, tm1=cap.tr_method.values == 1,
             hiP=p0 > 8, val_exps=cap._src.values == 'validation')
    res = {}
    for v in VARIANTS:
        P = np.zeros((len(cap), 6)); ok = True
        for e in E.index:
            p = OUT / 'pred' / f'{v}__{e}.npy'
            if not p.exists():
                ok = False; break
            P[cap.experiment_id.values == e] = np.load(p)
        if not ok:
            log(f'{v}: 미완료'); continue
        for al in (0.1, 0.0):
            lab = SM.viterbi(cap, (1 - al) * P + al * V)
            r = {g: float(f1_score(y[m], lab[m], labels=LAB, average='macro')) for g, m in G.items()}
            r['per_class'] = f1_score(y, lab, labels=LAB, average=None).round(4).tolist()
            res[f'{v}@a{al}'] = r
            log(f'{v:10s} α={al:<4} ' + ' | '.join(f'{g} {s:.4f}' for g, s in r.items() if g != 'per_class') + f' | {r["per_class"]}')
    (OUT / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    {'run': run, 'score': score}[sys.argv[1]]()
