"""프리즈 DINOv2 임베딩의 PCA 성분을 센서 인과 피처 생성기의 추가 신호로 넣어 OOF 5-fold(fast)로 센서 단독과 비교. 이미지 학습 없음.
PCA 는 라벨 행에서만 적합(비지도). 모두 현재·과거 프레임의 값만 사용(각 신호가 누적·창 피처를 받음)."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
from fusion_lib import emb_columns
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7)); BASE0 = list(F.BASE_SIGNALS)
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; cls = fr.bat_class.values
ids = tb.sample_id.astype(str).values
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
base_fs = None
for nm, spec, k in [("센서 단독", None, 0), ("b08 PCA4", "dinov2_b:b08_mean", 4), ("b08 PCA8", "dinov2_b:b08_mean", 8), ("b08 PCA16", "dinov2_b:b08_mean", 16), ("b05 PCA8", "dinov2_b:b05_mean", 8)]:
    d = raw.copy()
    extra = []
    if spec:
        E = emb_columns(ids, lab, spec, k, 0); extra = list(E.columns)
        for c in extra: d[c] = E[c].values
    F.BASE_SIGNALS = BASE0 + extra; F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    X = F.build_features(d).iloc[lab_i].reset_index(drop=True)
    O = np.zeros((len(y), 6))
    for kf in range(5):
        va = fold == kf
        m = F.build_model(); m.set_params(n_jobs=16, n_estimators=250, learning_rate=0.06)
        O[va] = m.fit(X[~va], y[~va]).predict_proba(X[va])
    p = F.postprocess(fr, O, 0.7); fs = np.array([f1(y[fold == kf], p[fold == kf]) for kf in range(5)])
    if base_fs is None: base_fs = fs; base_f = f1(y, p)
    print(f"{nm:10s} feat {X.shape[1]} | OOF {f1(y, p):.4f} ({f1(y, p) - base_f:+.4f}) 이긴 fold {int((fs > base_fs).sum())}/5 | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]}", flush=True)
