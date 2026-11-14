"""SOTA 0.8812 파이프라인을 우리 평가 규약으로 재측정: CV(Train GroupKFold(5, seed 0), 평가 fold 500행 캡) + Val(Train 학습).

모델·피처·디코딩은 sota_0.8812 코드 그대로:
  LightGBM A(rich 166) / 과충전 행은 C(압력 절대값·v_peak_ratio 제외), EfficientNet-B0 차이 입력, α=0.06, 인과적 단조 Viterbi.
차이: 이미지 seed 는 42 하나 (SOTA 최종은 42·7 평균). 모든 모델 체크포인트 저장·재사용.

  python sota_eval.py sensor     # LightGBM fold/Val 모델 → 확률
  python sota_eval.py vision     # EfficientNet fold/Val 모델 → 확률
  python sota_eval.py merge      # 결합·디코딩·점수
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

SOTA = Path('/home/nxcolelxu/Team10-uni-dthon-6th-battery-thermal-runaway/sota_0.8812')
sys.path.insert(0, str(SOTA))
import features as FT  # noqa: E402  (SOTA features.py)
import model as SM  # noqa: E402     (SOTA model.py)

D = Path('/home/nxcolelxu/unid/data/battery_data_light')
OUT = Path(__file__).resolve().parent / 'sota_eval'
(OUT / 'ckpt').mkdir(parents=True, exist_ok=True)
LAB = np.array([1, 2, 3, 4, 5, 6])
ALPHA = 0.06
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
    return pd.concat(parts).sort_index()


def frames():
    tr = pd.read_csv(D / 'train' / 'train.csv'); va = pd.read_csv(D / 'validation' / 'validation.csv')
    tr_cap = cap_rows(tr).reset_index(drop=True); va_cap = cap_rows(va).reset_index(drop=True)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    # 평가 단위: (이름, 학습 행 인덱스, 평가 프레임)
    units = []
    for k, (a, b) in enumerate(splits):
        ev = tr_cap[~tr_cap.experiment_id.isin(set(tr.experiment_id.iloc[a]))].reset_index(drop=True)
        units.append((f'fold{k}', a, ev))
    units.append(('val', np.arange(len(tr)), va_cap))
    return tr, va, tr_cap, va_cap, units


def sensor():
    import joblib, lightgbm as lgb
    tr, va, tr_cap, va_cap, units = frames()
    Xtr = FT.build(tr, 'rich')
    cols = list(Xtr.columns); cols_c = SM.cols_without_abs_pressure(cols)
    log(f'features {len(cols)} / overcharge model {len(cols_c)}')
    y = tr.label.values.astype(int)
    for name, a, ev in units:
        p = OUT / f'sensor_{name}.npy'
        if p.exists():
            continue
        ck = OUT / 'ckpt' / f'lgbm_{name}.joblib'
        if ck.exists():
            M = joblib.load(ck)
        else:
            params = dict(SM.LGB_PARAMS, n_jobs=6)
            M = dict(A=lgb.LGBMClassifier(**params).fit(Xtr.iloc[a], y[a]),
                     C=lgb.LGBMClassifier(**params).fit(Xtr.iloc[a][cols_c], y[a]))
            joblib.dump(M, ck)
        Xe = FT.build(ev, 'rich')[cols]
        P = M['A'].predict_proba(Xe)
        t2 = ev.tr_method.values == 2
        if t2.any():
            P[t2] = M['C'].predict_proba(Xe[cols_c].iloc[np.where(t2)[0]])
        np.save(p, P)
        log(f'sensor {name} done ({len(ev)} rows)')


def vision():
    import torch
    tr, va, tr_cap, va_cap, units = frames()
    dev = 'cuda'
    img_tr = SM.load_images(tr, D / 'train')
    img_va = SM.load_images(va, D / 'validation')
    log(f'images loaded {img_tr.shape} {img_va.shape}')
    sid_tr = pd.Series(np.arange(len(tr)), index=tr.sample_id.values)
    sid_va = pd.Series(np.arange(len(va)), index=va.sample_id.values)
    for name, a, ev in units:
        p = OUT / f'vision_{name}.npy'
        if p.exists():
            continue
        ck = OUT / 'ckpt' / f'vision_{name}_seed42.pt'
        net = SM.VisionNet(pretrained=not ck.exists()).to(dev)
        if ck.exists():
            net.load_state_dict(torch.load(ck, map_location=dev))
        else:
            df_a = tr.iloc[a].reset_index(drop=True)
            from train import train_vision          # SOTA train.py (sys.path 맨 앞이 sota_0.8812)
            net = train_vision(df_a, img_tr[a], 42, dev)
            torch.save(net.state_dict(), ck)
        # 평가 프레임(캡) 안에서 첫/15초/60초 전 프레임 인덱스 (Test 조건)
        src = (sid_va if name == 'val' else sid_tr).loc[ev.sample_id.values].to_numpy()
        I = (img_va if name == 'val' else img_tr)[src]
        f0, p15, p60 = SM.first_index(ev), SM.past_index(ev, 15), SM.past_index(ev, 60)
        net.eval(); out = []
        with torch.no_grad():
            for b in range(0, len(ev), 256):
                r = np.arange(b, min(b + 256, len(ev)))
                out.append(torch.softmax(net(SM.vision_input(I, r, f0[r], p15[r], p60[r], dev)).float(), 1).cpu().numpy())
        np.save(p, np.concatenate(out))
        log(f'vision {name} done')
        del net; torch.cuda.empty_cache()


def merge():
    tr, va, tr_cap, va_cap, units = frames()
    res = {}

    def sc(fr, P):
        lab = SM.viterbi(fr, P); y = fr.label.values.astype(int); oc = fr.tr_method.values == 2
        f = lambda m: float(f1_score(y[m], lab[m], labels=LAB, average='macro'))
        return dict(all=f(np.ones(len(y), bool)), overcharge=f(oc),
                    per_class=f1_score(y, lab, labels=LAB, average=None).round(4).tolist())
    have_img = all((OUT / f'vision_{n}.npy').exists() for n, _, _ in units)
    for tag, use_img in (('sensor_only', False),) + ((('sota_alpha0.06', True),) if have_img else ()):
        cv_fr, cv_P = [], []
        for name, a, ev in units:
            P = np.load(OUT / f'sensor_{name}.npy')
            if use_img:
                P = (1 - ALPHA) * P + ALPHA * np.load(OUT / f'vision_{name}.npy')
            if name == 'val':
                res.setdefault(tag, {})['val'] = sc(ev, P)
            else:
                cv_fr.append(ev); cv_P.append(P)
        res[tag]['cv'] = sc(pd.concat(cv_fr, ignore_index=True), np.concatenate(cv_P))
        r = res[tag]
        log(f'{tag:15s} CV {r["cv"]["all"]:.4f} OC {r["cv"]["overcharge"]:.4f} | Val {r["val"]["all"]:.4f} OC {r["val"]["overcharge"]:.4f} '
            f'| CV {r["cv"]["per_class"]} | Val {r["val"]["per_class"]}')
    (OUT / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    {'sensor': sensor, 'vision': vision, 'merge': merge}[sys.argv[1]]()
