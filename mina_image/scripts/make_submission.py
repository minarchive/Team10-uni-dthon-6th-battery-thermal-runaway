"""Train+Validation 전체로 학습해 Test 예측 -> 제출 CSV 생성 (시계열만 / 시계열+이미지 확률).

시계열만: ts_model/features.py 의 피처·모델·확신 cummax 후처리를 그대로 사용(= train.py --train_val + predict.py 와 같은 구성)
+이미지:  이미지 확률에서 만든 신호(img_exp, img_p456)를 SIGNALS 에 추가. Test 의 이미지 확률은 fold 모델 평균(*_test.npy),
          학습 행의 이미지 확률은 OOF. (분포가 약간 다르다 -> fusion_cv 에서 이득이 확인된 경우에만 사용할 것)
--aug:    학습 때 긴(>500행) 실험을 10/15초/혼합으로 다시 뽑은 버전을 추가(fusion_cv.py 와 동일)
검증: 제출 형식(행 수·순서·라벨 1~6), 실험별 단계 단조성, 라벨 분포, --compare 로 다른 제출과의 일치율
사용:
  python scripts/make_submission.py --ts_dir ts_model --name ts_only
  python scripts/make_submission.py --ts_dir ts_model --models ft_dinov2_b --name ts_img_dinov2b --compare outputs/submissions/ts_only.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--ts_dir", type=Path, required=True)
ap.add_argument("--data", type=Path, default=DATA)
ap.add_argument("--models", nargs="+", default=[])
ap.add_argument("--probs", default="image_probs")
ap.add_argument("--aug", action="store_true")
ap.add_argument("--aug_copies", type=int, default=1)
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--name", required=True)
ap.add_argument("--compare", type=Path, default=None)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, str(a.ts_dir))
import features as F  # 시계열 팀 코드

L = list(range(1, 7))
BASE0 = list(F.BASE_SIGNALS)
rng = np.random.default_rng(a.seed)


def read(split):
    d = pd.read_csv(a.data / split / f"{split}.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    d["sample_id"], d["experiment_id"] = d.sample_id.astype(str), d.experiment_id.astype(str)
    d["split"] = split
    return d


lab = pd.concat([read("train"), read("validation")], ignore_index=True)
tst = read("test")
use_img = bool(a.models)
if use_img:
    img = pd.read_parquet(ROOT / "outputs" / "preds" / f"{a.probs}.parquet")
    img["sample_id"] = img.sample_id.astype(str)
    for k in L:
        img[f"img_p{k}"] = np.mean([img[f"{m}_p{k}"].values for m in a.models], axis=0)
    P = img[[f"img_p{k}" for k in L]].values
    img["img_exp"], img["img_p456"] = P @ np.array(L), P[:, 3:].sum(1)
    cols = ["sample_id", "img_exp", "img_p456"]
    lab = lab.merge(img[cols], on="sample_id", how="left")
    tst = tst.merge(img[cols], on="sample_id", how="left")
    assert lab[["img_exp", "img_p456"]].notna().all().all(), "학습 행에 이미지 OOF 확률이 없음"
    assert tst[["img_exp", "img_p456"]].notna().all().all(), (
        "Test 행에 이미지 확률이 없음: 해당 모델의 *_test.npy 가 필요합니다(새 finetune.py 로 다시 실행해 생성)")
print(f"학습 행 {len(lab)} ({lab.experiment_id.nunique()}개 실험), Test {len(tst)}행 ({tst.experiment_id.nunique()}개 실험), 이미지={use_img}", flush=True)


def feats(frame):
    F.BASE_SIGNALS = BASE0 + (["img_exp", "img_p456"] if use_img else [])
    F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    return F.build_features(frame.reset_index(drop=True))


def subsample_idx(n, pat):
    if pat in ("10", "15"):
        g = int(pat) // 5
        return np.arange(int(rng.integers(0, g)), n, g)
    pos = [int(rng.integers(0, 3))]
    while True:
        nxt = pos[-1] + int(rng.choice([2, 3]))
        if nxt >= n:
            break
        pos.append(nxt)
    return np.array(pos)


Xtr = feats(lab)
cols = list(Xtr.columns)
Xs, ys = [Xtr], [lab.label.values]
if a.aug:
    for pat in ("10", "15", "mix"):
        for _ in range(a.aug_copies):
            parts = []
            for _, e in lab.groupby("experiment_id", sort=False):
                e = e.sort_values("relative_time_s", kind="stable")
                if len(e) > 500:
                    parts.append(e.iloc[subsample_idx(len(e), pat)])
            fr = pd.concat(parts).reset_index(drop=True)
            Xs.append(feats(fr)[cols])
            ys.append(fr.label.values)
    print("증강 포함 학습 행:", sum(len(y) for y in ys), flush=True)
X, y = pd.concat(Xs, ignore_index=True), np.concatenate(ys).astype(int)
model = F.build_model().fit(X, y)
Xte = feats(tst)[cols]
Pte = model.predict_proba(Xte)
pred = F.postprocess(tst.reset_index(drop=True), Pte, a.tau)

sub_order = pd.read_csv(a.data / "sample_submission.csv", dtype={"sample_id": "string"})["sample_id"].astype(str)
assert set(sub_order) == set(tst.sample_id), "sample_submission 과 test.csv 의 ID 가 다릅니다"
s = pd.Series(pred.astype(int), index=tst.sample_id.values)
sub = pd.DataFrame({"sample_id": sub_order.tolist(), "label": s.loc[sub_order.tolist()].values})
assert len(sub) == len(sub_order) == 3451 and sub.label.isin(L).all() and sub.sample_id.is_unique
out = ROOT / "outputs" / "submissions" / f"{a.name}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(out, index=False)
np.save(out.with_suffix(".proba.npy"), Pte.astype(np.float32))
print("저장:", out, len(sub), "행")

tt = tst.assign(label_pred=pred).sort_values(["experiment_id", "relative_time_s"], kind="stable")
mono = tt.groupby("experiment_id").label_pred.apply(lambda v: bool((np.diff(v.values) >= 0).all()))
print("단조 비감소 실험:", int(mono.sum()), "/", len(mono))
print("라벨 분포:", sub.label.value_counts(normalize=True).sort_index().round(3).to_dict())
print("실험별 단계 도달(최대 라벨):", tt.groupby("experiment_id").label_pred.max().value_counts().sort_index().to_dict())
if a.compare and a.compare.exists():
    o = pd.read_csv(a.compare, dtype={"sample_id": "string"}).set_index("sample_id").label
    agree = (o.loc[sub.sample_id.values].values == sub.label.values).mean()
    print(f"{a.compare.name} 과 일치율 {agree:.4f} (불일치 {int((1 - agree) * len(sub))}행)")
