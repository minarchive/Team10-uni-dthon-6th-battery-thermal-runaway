"""부드러운 단조 후처리 비교 + 혼동행렬.

1) all+balanced LightGBM으로 확률을 만든다.
   - Train: GroupKFold(5) out-of-fold (검증 fold는 500행 캡 적용)
   - Val  : Train 전체로 학습 후 500행 캡 Validation 예측
   → probs_<tag>.npz 로 저장 (후처리 실험은 이 파일만으로 반복 가능)
2) 후처리 여러 가지를 CV(OOF)와 Val 양쪽에서 Macro F1로 비교.
모든 후처리는 실험별로 시간순, 현재까지의 예측만 사용(인과적).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

import ts_model as M

L = np.array(M.LABELS)


def f1(y, p):
    return f1_score(y, p, labels=M.LABELS, average='macro', zero_division=0)


def load(d, split):
    return pd.read_csv(Path(d) / split / f'{split}.csv',
                       dtype={'sample_id': 'string', 'experiment_id': 'string'}).reset_index(drop=True)


def make_probs(tr, va, feature_fn=M.build_features, cw='balanced'):
    """OOF(train, 캡) + Val(캡) 확률. 반환 frame들은 시간순 정렬."""
    oof_frames, oof_P = [], []
    for a, b in GroupKFold(5).split(tr, groups=tr.experiment_id):
        m = M.build_model('lgbm', cw).fit(feature_fn(tr.iloc[a]), tr.label.iloc[a])
        ev = M.cap_rows(tr.iloc[b])
        oof_frames.append(ev); oof_P.append(m.predict_proba(feature_fn(ev)))
    m = M.build_model('lgbm', cw).fit(feature_fn(tr), tr.label)
    vc = M.cap_rows(va)
    return (pd.concat(oof_frames), np.vstack(oof_P)), (vc, m.predict_proba(feature_fn(vc))), m


# ---------------------------------------------------------------- 후처리
def _per_exp(frame, P, fn):
    """실험별 시간순으로 fn(P_exp) -> label 배열 적용, 원래 순서로 반환."""
    out = np.empty(len(frame), dtype=int)
    t = frame['relative_time_s'].to_numpy(float)
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        out[idx] = fn(P[idx], t[idx])
    return out


def ema(P, t, halflife_s):
    """시간 기준 지수이동평균(과거만)."""
    out = np.empty_like(P); out[0] = P[0]
    for i in range(1, len(P)):
        a = 1 - 0.5 ** ((t[i] - t[i - 1]) / halflife_s)
        out[i] = (1 - a) * out[i - 1] + a * P[i]
    return out


def ordinal_cummax(P, thr=0.5):
    """누적확률 c_s = P(단계>=s)의 누적최댓값 → c_s>thr 인 최대 s."""
    C = np.cumsum(P[:, ::-1], axis=1)[:, ::-1]           # C[:, s-1] = P(y >= s)
    C = np.maximum.accumulate(C, axis=0)
    return np.array([L[(c > thr)].max() if (c > thr).any() else 1 for c in C])


def conf_cummax(P, tau=0.7):
    """확신 조건 cummax: 확률 >= tau 인 예측만 '확정 단계'로 누적.
    출력 = max(확정 단계, 현재 예측) → 한 번 튄 오예측은 그 행에만 남고 이후로 전파되지 않음."""
    pred, conf = L[P.argmax(1)], P.max(1)
    committed = np.maximum.accumulate(np.where(conf >= tau, pred, 1))
    return np.maximum(committed, pred)


def kconsec(pred, k):
    """현재 상태보다 높은 예측이 k번 연속일 때만 올림 = cummax(rolling_min_k(pred))."""
    rm = pd.Series(pred).rolling(k, min_periods=1).min().to_numpy()
    return np.maximum.accumulate(rm).astype(int)


def methods():
    am = lambda P: L[P.argmax(1)]
    ms = {
        'raw': lambda P, t: am(P),
        'cummax': lambda P, t: np.maximum.accumulate(am(P)),
    }
    for h in [15, 30, 60]:
        ms[f'ema{h}'] = lambda P, t, h=h: am(ema(P, t, h))
        ms[f'ema{h}+cummax'] = lambda P, t, h=h: np.maximum.accumulate(am(ema(P, t, h)))
    for k in [2, 3, 5, 8]:
        ms[f'k{k}'] = lambda P, t, k=k: kconsec(am(P), k)
    for h in [15, 30]:
        for k in [2, 3]:
            ms[f'ema{h}+k{k}'] = lambda P, t, h=h, k=k: kconsec(am(ema(P, t, h)), k)
    for tau in [0.6, 0.7, 0.8]:
        ms[f'conf{tau}'] = lambda P, t, tau=tau: conf_cummax(P, tau)
    ms['ordinal'] = lambda P, t: ordinal_cummax(P)
    for h in [15, 30]:
        ms[f'ema{h}+ordinal'] = lambda P, t, h=h: ordinal_cummax(ema(P, t, h))
    return ms


def compare(oof, val):
    rows = []
    for name, fn in methods().items():
        po, pv = _per_exp(oof[0], oof[1], fn), _per_exp(val[0], val[1], fn)
        rows.append(dict(method=name, cv=f1(oof[0].label, po), val=f1(val[0].label, pv),
                         cv_c2=f1_score(oof[0].label, po, labels=[2], average=None)[0],
                         val_c2=f1_score(val[0].label, pv, labels=[2], average=None)[0]))
    df = pd.DataFrame(rows)
    base = df.set_index('method').loc['raw']
    df['d_cv'] = df.cv - base.cv; df['d_val'] = df.val - base.val; df['mean'] = (df.cv + df.val) / 2
    return df.sort_values('mean', ascending=False)


def report_confusion(frame, P, title):
    p = L[P.argmax(1)]
    cm = confusion_matrix(frame.label, p, labels=M.LABELS)
    print(f'\n[{title}] 혼동행렬 (행=정답, 열=예측)')
    print(pd.DataFrame(cm, index=[f'y{c}' for c in L], columns=[f'p{c}' for c in L]).to_string())
    print('per-class F1', f1_score(frame.label, p, labels=M.LABELS, average=None).round(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='/home/ubuntu/data/battery_data_light')
    ap.add_argument('--tag', default='v2')
    args = ap.parse_args()
    tr, va = load(args.data_dir, 'train'), load(args.data_dir, 'validation')
    cache = Path(f'probs_{args.tag}.npz')
    if cache.exists():   # 저장된 확률 재사용 (frame 인덱스는 원본 tr/va 인덱스)
        z = np.load(cache)
        oof, val = (tr.loc[z['oof_idx']], z['oof_P']), (va.loc[z['val_idx']], z['val_P'])
    else:
        oof, val, _ = make_probs(tr, va)
        np.savez(cache, oof_P=oof[1], oof_idx=oof[0].index.values, val_P=val[1], val_idx=val[0].index.values)
    report_confusion(*oof, 'CV OOF'); report_confusion(*val, 'Val')
    pd.set_option('display.width', 200)
    print('\n', compare(oof, val).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
