"""센서(시계열) LightGBM 의 5-fold OOF 확률 + Test 확률(fold 모델 평균) 저장. 이미지 OOF 와 같은 data/folds.parquet fold.

산출(folds.parquet 행 순서 기준, build_image_features 와 같은 정렬):
  outputs/preds/sensor_lgb_oof.npy   (라벨 행 수, 6)
  outputs/preds/sensor_lgb_test.npy  (test 행 수, 6)
fold 별로 끝날 때마다 outputs/preds/_sensor_fold{f}.npz 로 저장해 중단 후 재개 가능.
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

from common import DATA, ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--ts_dir", default=str(ROOT / "ts_model"))
ap.add_argument("--n_jobs", type=int, default=8)
ap.add_argument("--fast", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.ts_dir)
import features as F

T0 = time.time()
log = lambda *m: print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *m, flush=True)

df = table()
frames = []
for s in ["train", "validation", "test"]:
    d = pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    d["sample_id"], d["experiment_id"] = d.sample_id.astype(str), d.experiment_id.astype(str)
    frames.append(d)
raw = pd.concat(frames, ignore_index=True).set_index("sample_id", drop=False).loc[df.sample_id.astype(str).values].reset_index(drop=True)
log("feature 계산", len(raw), "행")
X = F.build_features(raw)
log("feature", X.shape)
y = df.label.values
fold = df.fold.values
lab = (y >= 1) & (fold >= 0)
te = np.where(y == -1)[0]
pos = np.cumsum(lab) - 1
oof = np.zeros((int(lab.sum()), 6), np.float32)
tsum = np.zeros((len(te), 6), np.float64)
out = ROOT / "outputs" / "preds"
for f in range(5):
    fp = out / f"_sensor_fold{f}.npz"
    va = np.where(lab & (fold == f))[0]
    if fp.exists():
        z = np.load(fp)
        oof[pos[va]], tsum = z["va"], tsum + z["te"]
        log(f"fold {f}: 재사용")
        continue
    tr = np.where(lab & (fold != f))[0]
    m = F.build_model()
    m.set_params(n_jobs=a.n_jobs)
    if a.fast:
        m.set_params(n_estimators=250, learning_rate=0.06)
    m.fit(X.iloc[tr], y[tr].astype(int))
    pv, pt = m.predict_proba(X.iloc[va]), m.predict_proba(X.iloc[te])
    np.savez(fp, va=pv, te=pt)
    oof[pos[va]], tsum = pv, tsum + pt
    log(f"fold {f}: 학습 행 {len(tr)}, 검증 {len(va)}")
np.save(out / "sensor_lgb_oof.npy", oof)
np.save(out / "sensor_lgb_test.npy", (tsum / 5).astype(np.float32))
from sklearn.metrics import f1_score
log("저장 완료. OOF 센서 argmax Macro F1 =", round(f1_score(y[lab], oof.argmax(1) + 1, average="macro"), 4))
