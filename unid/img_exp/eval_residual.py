"""센서-잔차 공동학습 출력(ft_res_*.npz의 cum = 융합 P(≥k)) vs 센서 단독 비교.

Train: OOF(캡, 확신 cummax), Val: 캡 + 실험 단위 bootstrap Δ.
    python eval_residual.py ft_res_dinov2s_u4.npz [ft_res_dinov2b_u4.npz ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import moe_gate as M

F, LAB = M.F, M.LAB
d = Path('/home/nxcolelxu/unid/data/battery_data_light')
tr, va = M.load_split(d, 'train'), M.load_split(d, 'validation')
zo, zv = np.load('seq_base_oof.npz'), np.load('seq_base_valtest.npz')
Ps_tr = pd.DataFrame(zo['probs'], index=zo['sample_id'].astype(str)).loc[tr.sample_id].to_numpy(float)
Ps_va = pd.DataFrame(zv['val_probs'], index=zv['val_sample_id'].astype(str)).loc[va.sample_id].to_numpy(float)
cap_tr, cap_va = F.cap_rows(tr).index.to_numpy(), F.cap_rows(va).index.to_numpy()
b_tr, b_va = F.postprocess(tr, Ps_tr, 0.7), F.postprocess(va, Ps_va, 0.7)
y_tr, y_va = tr.label.to_numpy(int), va.label.to_numpy(int)
pc = lambda y, p, c: f1_score(y[c], p[c], labels=LAB, average=None).round(3).tolist()
print(f'sensor  CV {M.score(tr, b_tr, cap_tr):.4f} {pc(y_tr, b_tr, cap_tr)}  VAL {M.score(va, b_va, cap_va):.4f} {pc(y_va, b_va, cap_va)}')
res = {}
for path in sys.argv[1:]:
    n = Path(path).stem
    P_tr = M.probs_of(M.load_img(path, tr.sample_id))
    P_va = M.probs_of(M.load_img(path, va.sample_id))
    l_tr, l_va = F.postprocess(tr, P_tr, 0.7), F.postprocess(va, P_va, 0.7)
    cv, vv = M.score(tr, l_tr, cap_tr), M.score(va, l_va, cap_va)
    bd_tr = M.boot_diff(tr, b_tr, l_tr, cap_tr, n=500)
    bd = M.boot_diff(va, b_va, l_va, cap_va)
    print(f'{n:22s} CV {cv:.4f} {pc(y_tr, l_tr, cap_tr)} Δ{bd_tr[0]:+.4f} [{bd_tr[1]:+.4f},{bd_tr[2]:+.4f}]')
    print(f'{"":22s} VAL {vv:.4f} {pc(y_va, l_va, cap_va)} Δ{bd[0]:+.4f} [{bd[1]:+.4f},{bd[2]:+.4f}] P(Δ>0)={bd[3]:.2f}')
    res[n] = dict(cv=cv, val=vv, cv_boot=bd_tr, val_boot=bd)
Path('res_residual.json').write_text(json.dumps(res, indent=1))
