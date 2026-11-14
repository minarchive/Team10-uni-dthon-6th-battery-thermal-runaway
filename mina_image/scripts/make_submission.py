"""Train+Validation 전체로 학습해 Test 예측 -> 제출 CSV 생성. fusion_cv.py 에서 이긴 변형을 같은 정의로 재현한다.

--variant ts      시계열만(ts_model/features.py 의 피처·모델·확신 cummax 후처리 그대로)
--variant light   + 이미지 신호 2개(img_exp, img_p456)           [--models 필요, *_test.npy 필요]
--variant probs6  + 이미지 확률 6개 + img_exp                    [--models 필요, *_test.npy 필요]
--variant emb     + 프리즈 임베딩 PCA 성분(--emb enc:layer --emb_k)  [test 행도 캐시에 있어 파인튜닝 test 예측 불필요]
--blend_w w       (선택) 학습한 시계열 모델 확률 P_ts 와 이미지 확률 P_img 를 (1-w)P_ts + w P_img 로 섞은 뒤 후처리 [--models 필요]
--aug             학습 때 긴(>500행) 실험을 10/15초/혼합으로 다시 뽑은 버전을 추가
--train_only      Train 만으로 학습(팀 val0.9195 모델과 같은 구성, 제출 파일 재현 확인용)
이미지 확률의 학습 행 값은 OOF, Test 행 값은 fold 모델 평균(*_test.npy)이라 분포가 약간 다르다 -> fusion_cv 에서 이득이 확인된 경우에만 사용.
검증: 제출 형식(행 수·순서·라벨 1~6), 실험별 단조성, 라벨 분포, --compare 로 다른 제출과의 일치율
사용 예:
  python scripts/make_submission.py --ts_dir ts_model --name ts_only
  python scripts/make_submission.py --ts_dir ts_model --variant light --models ft_dinov2_b --name light_dinov2b --compare outputs/submissions/ts_only.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, ROOT
from fusion_lib import IMG_VARIANTS, add_image_cols, emb_columns

ap = argparse.ArgumentParser()
ap.add_argument("--ts_dir", type=Path, required=True)
ap.add_argument("--data", type=Path, default=DATA)
ap.add_argument("--variant", default="ts", choices=["ts", "light", "probs6", "emb"])
ap.add_argument("--models", nargs="+", default=[])
ap.add_argument("--probs", default="image_probs")
ap.add_argument("--emb", default="")
ap.add_argument("--emb_k", type=int, default=8)
ap.add_argument("--blend_w", type=float, default=0.0)
ap.add_argument("--aug", action="store_true")
ap.add_argument("--aug_copies", type=int, default=1)
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--name", required=True)
ap.add_argument("--compare", type=Path, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--train_only", action="store_true")
a = ap.parse_args()
sys.path.insert(0, str(a.ts_dir))
import features as F  # 시계열 팀 코드

L = list(range(1, 7))
BASE0 = list(F.BASE_SIGNALS)
rng = np.random.default_rng(a.seed)
need_img = a.variant in IMG_VARIANTS or a.blend_w > 0
if need_img:
    assert a.models, "이 설정은 --models 가 필요합니다"
if a.variant == "emb":
    assert a.emb, "emb 는 --emb enc:layer 가 필요합니다"


def read(split):
    d = pd.read_csv(a.data / split / f"{split}.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    d["sample_id"], d["experiment_id"] = d.sample_id.astype(str), d.experiment_id.astype(str)
    d["split"] = split
    return d


lab = pd.concat([read("train")] + ([] if a.train_only else [read("validation")]), ignore_index=True)
tst = read("test")
if need_img:
    lab, tst = add_image_cols(lab, a.probs, a.models), add_image_cols(tst, a.probs, a.models)
    assert lab["img_exp"].notna().all(), "학습 행에 이미지 OOF 확률이 없음"
    assert tst["img_exp"].notna().all(), "Test 행에 이미지 확률이 없음: 해당 모델의 *_test.npy 가 필요합니다(새 finetune.py 로 다시 실행해 생성)"
if a.variant == "emb":
    ids = np.concatenate([lab.sample_id.values, tst.sample_id.values])
    E = emb_columns(ids, np.arange(len(ids)) < len(lab), a.emb, a.emb_k, a.seed)
    lab = pd.concat([lab.reset_index(drop=True), E.iloc[:len(lab)].reset_index(drop=True)], axis=1)
    tst = pd.concat([tst.reset_index(drop=True), E.iloc[len(lab):].reset_index(drop=True)], axis=1)
extra = IMG_VARIANTS.get(a.variant, [f"emb_{i}" for i in range(a.emb_k)] if a.variant == "emb" else [])
print(f"학습 행 {len(lab)} ({lab.experiment_id.nunique()}개 실험), Test {len(tst)}행 ({tst.experiment_id.nunique()}개 실험) | variant={a.variant} blend_w={a.blend_w}", flush=True)


def feats(frame):
    F.BASE_SIGNALS = BASE0 + extra
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
tst = tst.reset_index(drop=True)
Pte = model.predict_proba(feats(tst)[cols])
if a.blend_w > 0:
    Pte = (1 - a.blend_w) * Pte + a.blend_w * tst[[f"img_p{k}" for k in L]].values
pred = F.postprocess(tst, Pte, a.tau)

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
print("단조 비감소 실험:", int(mono.sum()), "/", len(mono), "(확신 cummax 후처리는 단조를 강제하지 않음)")
print("라벨 분포:", sub.label.value_counts(normalize=True).sort_index().round(3).to_dict())
if a.compare and a.compare.exists():
    o = pd.read_csv(a.compare, dtype={"sample_id": "string"}).set_index("sample_id").label
    agree = (o.loc[sub.sample_id.values].values == sub.label.values).mean()
    print(f"{a.compare.name} 과 일치율 {agree:.4f} (불일치 {int(round((1 - agree) * len(sub)))}행)")
