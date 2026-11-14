"""GBM 백엔드 속도·점수 비교 (한 fold): lgbm_cpu vs lgbm_cuda vs xgb_cuda.  GPU 노드에서 실행."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402
import exp_img as E  # noqa: E402

tr = E.load(sys.argv[1], 'train')
X = F.build_features(tr)
y = tr.label.astype(int).to_numpy()
a, b = next(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
fev = F.cap_rows(tr.iloc[b])
Xtr, ytr, Xev, yev = X.iloc[a], y[a], X.loc[fev.sample_id.values], fev.label.astype(int).to_numpy()
print(f'train {Xtr.shape} eval {Xev.shape}', flush=True)
for backend in ['lgbm_cpu', 'lgbm_cuda', 'xgb_cuda']:
    os.environ['GBM_BACKEND'] = backend
    try:
        t0 = time.time()
        m = E.lgbm(0).fit(Xtr, ytr)
        t1 = time.time() - t0
        P = m.predict_proba(Xev)
        f = f1_score(yev, F.LABELS[P.argmax(1)], labels=F.LABELS, average='macro')
        print(f'{backend:10s} fit {t1:6.1f}s  macroF1 {f:.4f}', flush=True)
    except Exception as e:
        print(f'{backend:10s} FAILED: {type(e).__name__}: {str(e)[:200]}', flush=True)
