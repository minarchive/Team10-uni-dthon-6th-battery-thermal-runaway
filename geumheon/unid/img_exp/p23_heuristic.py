"""2→3 전환 휴리스틱 시험 (oc4 R_J6wide + ConvNeXt 0.1 확률에 후처리로 적용).
H0: t1 누적최대 < 50°C 이면 3단계 이상 금지 (Train+Val 97개 실험 모두 2→3 시점 t1 ≥ 50.1°C)
H1: t1 누적최대 ≥ 50°C 인 행에서 2단계 확률을 b 로 나눔 (3단계 이상으로 기울이기)"""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from final_oc import F, load, D
from final_oc2 import cnn

def t1cmax(fr):
    t = fr.relative_time_s.to_numpy(float); x = pd.to_numeric(fr.meas_t1, errors='coerce').to_numpy(float); out = np.empty(len(fr))
    for _, idx in fr.reset_index(drop=True).groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]; out[idx] = np.fmax.accumulate(np.nan_to_num(x[idx], nan=-np.inf))
    return out

def apply(P, c, h0, b):
    P = P.copy()
    if h0: P[c < 50, 2:] = 0                       # 3~6단계 금지
    hi = c >= 50; P[hi, 1] /= b
    return P / P.sum(1, keepdims=True)

def lag3(fr, lab):
    out = []
    for _, g in fr.assign(p=lab).groupby('experiment_id'):
        g = g.sort_values('relative_time_s'); t = g.relative_time_s.values
        a = np.where(g.p.values >= 3)[0]; b_ = np.where(g.label.values >= 3)[0]
        if len(a) and len(b_): out.append(t[a[0]] - t[b_[0]])
    return np.median(out), np.mean(out)

def ev(fr, P):
    lab = F.postprocess(fr, P, 0.7); y = fr.label.astype(int).to_numpy(); oc = (fr.tr_method == 2).to_numpy()
    f = lambda m: f1_score(y[m], lab[m], labels=[1, 2, 3, 4, 5, 6], average='macro')
    pc = f1_score(y, lab, labels=[1, 2, 3, 4, 5, 6], average=None)
    return f(np.ones(len(y), bool)), f(oc), pc[1], pc[2], *lag3(fr, lab)

if __name__ == '__main__':
    tr, va = [load(D, s).reset_index(drop=True) for s in ('train', 'validation')]
    for f in (tr, va): f['sample_id'] = f.sample_id.astype(str)
    tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)
    H = 'final_oc4_gpu/'
    P_cv = 0.9 * np.load(H + 'oof_R_J6wide.npy') + 0.1 * cnn('ft_convnext_tiny.npz', tr_cap.sample_id.values)
    P_va = 0.9 * np.load(H + 'valprobs_R_J6wide.npy') + 0.1 * cnn('ft_convnext_tiny.npz', va_cap.sample_id.values)
    c_cv, c_va = t1cmax(tr_cap), t1cmax(va_cap)
    # 규칙 검증: 라벨 ≥3 인 행은 모두 t1 누적최대 ≥ 50 인가
    for nm, fr, c in (('tr_cap', tr_cap, c_cv), ('va_cap', va_cap, c_va)):
        y = fr.label.astype(int).to_numpy(); print(f'{nm}: label>=3 행 중 t1cmax<50 비율 {(c[y >= 3] < 50).mean():.4f}')
    print(f'\n{"설정":16s} | {"CV":>6s} {"CV_OC":>6s} {"F1_2":>6s} {"F1_3":>6s} {"지연중앙":>7s} {"지연평균":>7s} | {"Val":>6s} {"Val_OC":>6s} {"F1_2":>6s} {"F1_3":>6s} {"지연중앙":>7s} {"지연평균":>7s}')
    for h0 in (False, True):
        for b in (1, 1.25, 1.5, 2, 3, 4):
            a = ev(tr_cap, apply(P_cv, c_cv, h0, b)); v = ev(va_cap, apply(P_va, c_va, h0, b))
            print(f'{("H0+" if h0 else "") + f"b={b}":16s} | ' + ' '.join(f'{x:6.4f}' if i < 4 else f'{x:+7.0f}' for i, x in enumerate(a)) + ' | ' + ' '.join(f'{x:6.4f}' if i < 4 else f'{x:+7.0f}' for i, x in enumerate(v)))
