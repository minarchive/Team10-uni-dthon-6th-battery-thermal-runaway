"""시계열(센서) 모델에 넘길 이미지 확률 테이블 생성.

각 모델 태그(예: ft_dino_r50, dino_r50__s4_gap)의 *_oof.npy(라벨 행) 와 *_test.npy(test 행, fold 모델 평균)를 합쳐
outputs/preds/image_probs.parquet 로 저장: sample_id, experiment_id, relative_time_s, split, label, fold, <tag>_p1..p6

주의: 센서 모델에서 쓸 때 반드시 같은 data/folds.parquet 의 fold 를 써야 한다(OOF 확률이 그 분할 기준이라
다른 분할을 쓰면 검증 fold 의 정보가 학습에 섞여 누수가 생긴다).
"""
import argparse

import numpy as np
import pandas as pd

from common import ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--out", default="image_probs")
a = ap.parse_args()
P = ROOT / "outputs" / "preds"
df = table()
lab = (df.label.values >= 1) & (df.fold.values >= 0)
te = df.label.values == -1
out = df[["sample_id", "experiment_id", "relative_time_s", "split", "label", "fold"]].copy()
for tag in a.models:
    oof, test = np.load(P / f"{tag}_oof.npy"), P / f"{tag}_test.npy"
    arr = np.full((len(df), 6), np.nan, np.float32)
    arr[lab] = oof
    if test.exists():
        arr[te] = np.load(test)
    else:
        print("test 예측 없음:", tag)
    for k in range(6):
        out[f"{tag}_p{k + 1}"] = arr[:, k]
out.to_parquet(P / f"{a.out}.parquet", index=False)
print("saved", P / f"{a.out}.parquet", out.shape)
print("NaN 비율(test 예측 없는 모델 확인):", out.filter(like="_p1").isna().mean().round(3).to_dict())
