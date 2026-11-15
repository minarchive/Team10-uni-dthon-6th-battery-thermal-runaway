"""LightGBM 래퍼 (WXGB 와 같은 인터페이스: fit(X, y, w_extra), predict_proba, classes_, .m).

pickle 로 저장된 체크포인트를 다른 스크립트에서도 불러올 수 있도록 별도 모듈에 둔다.
"""
import os

import numpy as np


class WLGB:
    """balanced 클래스 가중치 × 행별 추가 가중치. 기본값은 XGBBalanced(잎 31, 500, lr 0.03) 와 맞춤."""

    def __init__(self, seed, **kw):
        import lightgbm as lgb
        p = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, max_depth=-1,
                 min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
                 reg_lambda=1.0, random_state=seed, n_jobs=int(os.environ.get('GBM_THREADS', '6')),
                 verbose=-1)
        p.update(kw)
        self.m = lgb.LGBMClassifier(**p)

    def fit(self, X, y, w_extra):
        y = np.asarray(y); self.classes_ = np.unique(y)
        cnt = np.array([(y == c).sum() for c in self.classes_], float)
        w = (len(y) / (len(self.classes_) * cnt))[np.searchsorted(self.classes_, y)] * w_extra
        self.m.fit(X.to_numpy(np.float32), np.searchsorted(self.classes_, y), sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X.to_numpy(np.float32) if hasattr(X, 'to_numpy') else X)
