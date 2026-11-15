"""oc4(R_J6wide) 피처 840개 중 Test 에서 값 분포가 달라지는 피처 찾기 (Test 라벨 미사용, 입력 분포만 비교).
frac_out: Test 행 중 Train+Val 범위 [min,max] 밖 비율 / ks: Train+Val(500행 캡) vs Test KS 통계량."""
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
import final_oc as O
from final_oc import F, load, D
import final_oc4 as Q4

tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
for f in (tr, va, te): f['sample_id'] = f.sample_id.astype(str)
tv = pd.concat([F.cap_rows(tr), F.cap_rows(va)], ignore_index=True)
Xa, Ha = Q4.feats_r(tv); Xt, Ht = Q4.feats_r(te)
Xa = Xa[Ha.to_numpy()]; Xt = Xt[Ht.to_numpy()]          # oc4 모델이 실제로 보는 45°C 이후 행만
rows = []
for c in Xa.columns:
    a = Xa[c].to_numpy(float); t = Xt[c].to_numpy(float); a = a[np.isfinite(a)]; t = t[np.isfinite(t)]
    if len(a) < 10 or len(t) < 10: continue
    lo, hi = a.min(), a.max()
    rows.append(dict(feat=c, frac_out=float(((t < lo) | (t > hi)).mean()), ks=float(ks_2samp(a, t).statistic),
                     tv_med=float(np.median(a)), te_med=float(np.median(t)), tv_min=lo, tv_max=hi, te_min=t.min(), te_max=t.max()))
R = pd.DataFrame(rows).sort_values('ks', ascending=False)
R.to_csv('ood_oc4.csv', index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 40)
print('rows after45: trval', len(Xa), 'test', len(Xt), '| feats', len(R))
print(R.head(45).round(3).to_string(index=False))
print('\nfrac_out top'); print(R.sort_values('frac_out', ascending=False).head(20).round(3).to_string(index=False))
