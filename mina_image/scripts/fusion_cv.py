"""시계열 모델(val0.9195, features.py)에 이미지 확률을 추가했을 때의 이득을 같은 조건으로 비교.

방식: 시계열 팀의 features.py 를 그대로 import 해서
  - 이미지 확률에서 만든 신호 2개(img_exp = 기대 단계, img_p456 = P(단계>=4))를 SIGNALS 에 추가
  - 그러면 누적/윈도우/임계 feature 가 같은 인과적 방식으로 이미지 신호에도 계산된다(각 신호당 약 35개 feature)
  - 학습은 전체 프레임(train.py 와 동일), 평가는 Test 와 같은 조건(500행 초과 실험 균등 다운샘플 + 초 기준 윈도우) + 확신 cummax 후처리
비교: baseline(시계열만) vs +이미지. 같은 fold 에서 짝지은 차이를 보고한다.
protocol:
  cv       = data/folds.parquet 의 5-fold(Train+Val 97개 실험, 이미지 OOF 와 같은 분할)
  trainval = Train 으로 학습 -> Validation 평가(시계열 팀 점수 0.9195 재현 확인용; 이미지 OOF 는 Val 실험을 제외한 fold 모델 출력이지만
             다른 Val 실험이 그 모델 학습에 쓰였으므로 엄밀한 Train-only 평가는 아님)
주의: 이미지 확률이 OOF 라 2단계(stacking) 평가에 약한 누수 가능성이 있다. 최종 채택 전 중첩 CV 로 한 번 더 확인할 것.
test 용 이미지 확률 = fold 모델 평균(분포가 OOF 보다 약간 좋음 -> 보수적으로 해석)
사용: python scripts/fusion_cv.py --ts_dir <features.py 가 있는 폴더> --models ft_dino_r50 [--protocol trainval]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import DATA, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--ts_dir", type=Path, required=True)
ap.add_argument("--models", nargs="+", required=True, help="image_probs.parquet 의 모델 태그(여러 개면 확률 평균)")
ap.add_argument("--probs", default="image_probs")
ap.add_argument("--protocol", default="cv", choices=["cv", "trainval"])
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--max_folds", type=int, default=5)
a = ap.parse_args()
sys.path.insert(0, str(a.ts_dir))
import features as F  # 시계열 팀 코드

L = list(range(1, 7))
BASE0 = list(F.BASE_SIGNALS)
img = pd.read_parquet(ROOT / "outputs" / "preds" / f"{a.probs}.parquet")
img["sample_id"] = img.sample_id.astype(str)
for k in L:
    img[f"img_p{k}"] = np.mean([img[f"{m}_p{k}"].values for m in a.models], axis=0)
P = img[[f"img_p{k}" for k in L]].values
img["img_exp"] = P @ np.array(L)
img["img_p456"] = P[:, 3:].sum(1)
sens = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}).assign(split=s)
                  for s in ["train", "validation"]], ignore_index=True)
sens["sample_id"] = sens.sample_id.astype(str)
sens["experiment_id"] = sens.experiment_id.astype(str)
df = sens.merge(img[["sample_id", "fold", "img_exp", "img_p456"]], on="sample_id", how="inner")
assert len(df) == len(sens), "image_probs 와 센서 CSV 행이 맞지 않음"
df = df.reset_index(drop=True)
print("rows", len(df), "experiments", df.experiment_id.nunique())


def feats(frame, use_img):
    F.BASE_SIGNALS = BASE0 + (["img_exp", "img_p456"] if use_img else [])
    F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    return F.build_features(frame.reset_index(drop=True))


cap = F.cap_rows(df).reset_index(drop=True)  # 평가 조건: Test 처럼 500행 상한
variants = {}
for name, use_img in [("시계열만", False), ("시계열+이미지", True)]:
    Xf, Xc = feats(df, use_img), feats(cap, use_img)
    variants[name] = (Xf, Xc[Xf.columns])
    print(f"{name}: feature {Xf.shape[1]}개", flush=True)

if a.protocol == "cv":
    cap_fold = cap.fold.values
    splits = [((df.fold != f).values, cap_fold == f) for f in range(a.max_folds)]
else:
    splits = [((df.split == "train").values, (cap.split == "validation").values)]

res = {n: dict(raw=[], post=[], adj=[], pred=[], true=[], bat=[], percls=[]) for n in variants}
for fi, (tr, va) in enumerate(splits):
    for name, (Xf, Xc) in variants.items():
        m = F.build_model().fit(Xf[tr], df.label.values[tr].astype(int))
        Pv = m.predict_proba(Xc[va])
        fr = cap[va].reset_index(drop=True)
        y = fr.label.astype(int).values
        raw = F.LABELS[Pv.argmax(1)]
        post = F.postprocess(fr, Pv, a.tau)
        r = res[name]
        r["raw"].append(f1_score(y, raw, labels=L, average="macro", zero_division=0))
        r["post"].append(f1_score(y, post, labels=L, average="macro", zero_division=0))
        r["adj"].append(float((np.abs(post - y) <= 1).mean()))
        r["pred"].append(post)
        r["true"].append(y)
        r["bat"].append(fr.bat_class.values)
        r["percls"].append(f1_score(y, post, labels=L, average=None, zero_division=0))
    print(f"fold {fi}: " + " | ".join(f"{n} post {res[n]['post'][-1]:.4f} (raw {res[n]['raw'][-1]:.4f})" for n in variants), flush=True)

rows = []
for n, r in res.items():
    yt, yp, bt = np.concatenate(r["true"]), np.concatenate(r["pred"]), np.concatenate(r["bat"])
    pc = np.mean(r["percls"], axis=0)
    print(f"\n== {n}: F1(후처리) {np.mean(r['post']):.4f} ± {np.std(r['post']):.4f} | 후처리 없음 {np.mean(r['raw']):.4f} | 인접 정확도 {np.mean(r['adj']):.4f}")
    print("   단계별 F1(후처리, fold 평균):", [round(float(v), 3) for v in pc])
    for b in (1, 2, 3):
        mk = bt == b
        print(f"   bat_class {b}: macro F1 {f1_score(yt[mk], yp[mk], labels=sorted(set(yt[mk])), average='macro', zero_division=0):.4f} (n={mk.sum()})")
    rows.append(dict(models="+".join(a.models), protocol=a.protocol, variant=n, f1_post=np.mean(r["post"]), f1_std=np.std(r["post"]),
                     f1_raw=np.mean(r["raw"]), adj=np.mean(r["adj"]), per_class=[round(float(v), 3) for v in pc]))
if len(splits) > 1:
    d = np.array(res["시계열+이미지"]["post"]) - np.array(res["시계열만"]["post"])
    print(f"\nfold별 차이(+이미지 - 시계열만): {np.round(d, 4)} 평균 {d.mean():+.4f} | 이미지가 이긴 fold {int((d > 0).sum())}/{len(d)}")
else:
    print(f"\n차이(+이미지 - 시계열만): {res['시계열+이미지']['post'][0] - res['시계열만']['post'][0]:+.4f}  (Validation 19개 실험 기준, ±0.02~0.03 변동 가능)")
p = ROOT / "outputs" / "logs" / "fusion_results.csv"
pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)
