"""2→3 전환 분석: (1) oc4 R_J6wide(+ConvNeXt 0.1) 예측이 얼마나 늦는지 (2) 정답 전환 시점의 센서 값 분포."""
import numpy as np, pandas as pd
from final_oc import F, load, D
from final_oc2 import cnn
tr, va = [load(D, s).reset_index(drop=True) for s in ('train', 'validation')]
for f in (tr, va): f['sample_id'] = f.sample_id.astype(str)
tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)
H = 'final_oc4_gpu/'
P_cv = 0.9 * np.load(H + 'oof_R_J6wide.npy') + 0.1 * cnn('ft_convnext_tiny.npz', tr_cap.sample_id.values)
P_va = 0.9 * np.load(H + 'valprobs_R_J6wide.npy') + 0.1 * cnn('ft_convnext_tiny.npz', va_cap.sample_id.values)

def onset(t, lab, L):
    i = np.where(lab >= L)[0]; return t[i[0]] if len(i) else np.nan

print('== (1) 예측 전환 시각 − 정답 전환 시각 (초, +면 늦음)')
for nm, fr, P in (('CV', tr_cap, P_cv), ('Val', va_cap, P_va)):
    pred = F.postprocess(fr, P, 0.7); rows = []
    for e, g in fr.assign(pred=pred).groupby('experiment_id'):
        g = g.sort_values('relative_time_s'); t = g.relative_time_s.values
        r = dict(e=e, tm=g.tr_method.iloc[0])
        for L in (2, 3, 4, 5, 6):
            r[f'lag{L}'] = onset(t, g.pred.values, L) - onset(t, g.label.values, L)
        rows.append(r)
    R = pd.DataFrame(rows)
    for tm in (1, 2):
        x = R[R.tm == tm]
        print(f' {nm} tm{tm} ({len(x)}개): ' + ' | '.join(f'{L}: 중앙 {x[f"lag{L}"].median():+.0f} 평균 {x[f"lag{L}"].mean():+.0f}' for L in (2, 3, 4, 5, 6)))
    x = R.lag3.dropna(); print(f'   {nm} 2→3 지연 분포 분위수:', x.quantile([.1, .25, .5, .75, .9]).round(0).tolist(), '늦음 비율', round((x > 0).mean(), 2))

print('\n== (2) 정답 2→3 시점의 센서 값 (Train+Val 원본 5초 간격, 실험별)')
tv = pd.concat([tr, va], ignore_index=True); rows = []
for e, g in tv.groupby('experiment_id'):
    g = g.sort_values('relative_time_s'); t = g.relative_time_s.values; lab = g.label.values
    t1 = g.meas_t1.astype(float).values; t2 = g.meas_t2.astype(float).values; p = g.meas_p.astype(float).values; v = g.meas_v.astype(float).values
    th = pd.to_numeric(g.thermal_t, errors='coerce').values
    if not ((lab == 2).any() and (lab == 3).any()): continue
    i2 = np.argmax(lab >= 2); i3 = np.argmax(lab >= 3)
    def rate(x, i, w=30):
        j = np.searchsorted(t, t[i] - w); return (x[i] - x[j]) / max(t[i] - t[j], 1)
    rows.append(dict(e=e, tm=g.tr_method.iloc[0], bat=g.bat_class.iloc[0],
        dur2=t[i3] - t[i2], t1_at3=t1[i3], t2_at3=t2[i3], tmax_at3=max(t1[i3], t2[i3]), th_at3=th[i3],
        t1_rate30=rate(t1, i3), t1_rate30_at2=rate(t1, i2), t1_rise_from2=t1[i3] - t1[i2],
        p_rise_from2=p[i3] - p[i2], p_rise_from0=p[i3] - p[:7].mean(), p_rate30=rate(p, i3),
        v_rel=v[i3] / v[:7].mean(), v_rel_peak=v[i3] / v[:i3 + 1].max()))
E = pd.DataFrame(rows); E.to_csv('p23_onset_signals.csv', index=False)
pd.set_option('display.width', 220)
for tm in (1, 2):
    x = E[E.tm == tm].drop(columns=['e', 'tm'])
    d = x.describe(percentiles=[.1, .5, .9]).T[['count', 'min', '10%', '50%', '90%', 'max']]
    d['cv'] = (x.std() / x.mean().abs()).round(2)
    print(f'\n tm{tm} ({len(x)}개)'); print(d.round(2).to_string())
