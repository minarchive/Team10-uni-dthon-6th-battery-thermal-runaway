"""시계열 모델(val0.9195, features.py)에 (1) 이미지 확률 신호 추가, (2) 샘플링 간격 증강을 적용했을 때의 효과를 같은 조건으로 비교.

배경: Train/Val 은 5초 고정인데 Test 의 긴 실험 2개(1,000행)는 10·15초가 섞여 있다. 학습 분포와 달라서 초 기준 윈도우 feature 의 성격이 달라진다.
  --aug : 학습 때 500행 초과(긴) 실험을 10초/15초/혼합{10,15}로 다시 뽑은 버전을 추가(시작 위치 무작위, --aug_copies 만큼). 짧은 실험은 Test 처럼 5초 유지(--aug_all 이면 전부)
평가는 두 가지를 같이 본다:
  cap500 = 시계열 팀 프로토콜(500행 초과 실험 균등 다운샘플)
  mix    = 500행 초과 실험을 10/15초 혼합으로 다시 뽑음(Test 긴 실험과 같은 간격 분포)
비교: baseline(시계열만) vs +이미지(img_exp, img_p456 을 SIGNALS 에 추가 -> 같은 인과적 feature 가 이미지 신호에도 계산됨)
protocol: cv = data/folds.parquet 5-fold(Train+Val 97개 실험, 이미지 OOF 와 동일 분할) / trainval = Train 학습 -> Validation 평가(0.9195 재현 확인용)
주의: 이미지 확률은 OOF 라 2단계 평가에 약한 누수 가능성이 있다. 채택 전 중첩 CV 로 재확인. test 용 이미지 확률(fold 평균)은 OOF 보다 분포가 약간 좋다.
사용: python scripts/fusion_cv.py --ts_dir <features.py 폴더> --models ft_dino_r50 [--protocol trainval] [--aug]
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
ap.add_argument("--models", nargs="+", default=[], help="image_probs.parquet 모델 태그(여러 개면 확률 평균). 비우면 시계열만")
ap.add_argument("--probs", default="image_probs")
ap.add_argument("--protocol", default="cv", choices=["cv", "trainval"])
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--max_folds", type=int, default=5)
ap.add_argument("--aug", action="store_true")
ap.add_argument("--aug_all", action="store_true", help="짧은 실험도 다시 뽑아 증강")
ap.add_argument("--aug_copies", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, str(a.ts_dir))
import features as F  # 시계열 팀 코드

L = list(range(1, 7))
BASE0 = list(F.BASE_SIGNALS)
rng = np.random.default_rng(a.seed)

sens = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}).assign(split=s)
                  for s in ["train", "validation"]], ignore_index=True)
sens["sample_id"] = sens.sample_id.astype(str)
sens["experiment_id"] = sens.experiment_id.astype(str)
variants = {"시계열만": False}
if a.models:
    img = pd.read_parquet(ROOT / "outputs" / "preds" / f"{a.probs}.parquet")
    img["sample_id"] = img.sample_id.astype(str)
    for k in L:
        img[f"img_p{k}"] = np.mean([img[f"{m}_p{k}"].values for m in a.models], axis=0)
    P = img[[f"img_p{k}" for k in L]].values
    img["img_exp"], img["img_p456"] = P @ np.array(L), P[:, 3:].sum(1)
    df = sens.merge(img[["sample_id", "fold", "img_exp", "img_p456"]], on="sample_id", how="inner")
    assert len(df) == len(sens), "image_probs 와 센서 CSV 행이 맞지 않음"
    variants["시계열+이미지"] = True
else:
    folds = pd.read_parquet(ROOT / "data" / "folds.parquet")[["sample_id", "fold"]]
    df = sens.merge(folds, on="sample_id", how="inner")
df = df.reset_index(drop=True)
print("rows", len(df), "experiments", df.experiment_id.nunique(), "| 변형:", list(variants), "| aug:", a.aug, flush=True)


def subsample_idx(n, pat):
    if pat == "5":
        return np.arange(n)
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


def resample(frame, pat, mode):
    """mode: 'train_long' 긴(>500행) 실험만 다시 뽑고 짧은 실험은 제외 / 'train_all' 전부 다시 뽑음 /
    'eval' 긴 실험만 다시 뽑고 짧은 실험은 5초 그대로 포함(Test 구성과 동일)."""
    parts = []
    for _, e in frame.groupby("experiment_id", sort=False):
        e = e.sort_values("relative_time_s", kind="stable")
        is_long = len(e) > 500
        if mode == "train_long" and not is_long:
            continue
        if mode == "eval" and not is_long:
            parts.append(e)
            continue
        parts.append(e.iloc[subsample_idx(len(e), pat)])
    return pd.concat(parts).reset_index(drop=True)


def feats(frame, use_img):
    F.BASE_SIGNALS = BASE0 + (["img_exp", "img_p456"] if use_img else [])
    F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    return F.build_features(frame.reset_index(drop=True))


aug_frames = []
if a.aug:
    for pat in ("10", "15", "mix"):
        for _ in range(a.aug_copies):
            aug_frames.append(resample(df, pat, "train_all" if a.aug_all else "train_long"))
    print("증강 프레임:", [len(f) for f in aug_frames], "(원본", len(df), ")", flush=True)
evals = {"cap500": F.cap_rows(df).reset_index(drop=True), "mix": resample(df, "mix", "eval")}

data = {}
for name, use_img in variants.items():
    Xf = feats(df, use_img)
    cols = list(Xf.columns)
    Xs, ys, fs, ss = [Xf], [df.label.values], [df.fold.values], [df.split.values]
    for fr in aug_frames:
        Xs.append(feats(fr, use_img)[cols]); ys.append(fr.label.values); fs.append(fr.fold.values); ss.append(fr.split.values)
    data[name] = dict(X=pd.concat(Xs, ignore_index=True), y=np.concatenate(ys).astype(int), fold=np.concatenate(fs), split=np.concatenate(ss),
                      ev={en: feats(fr, use_img)[cols] for en, fr in evals.items()})
    print(f"{name}: feature {len(cols)}개, 학습 행 {len(data[name]['y'])}", flush=True)

n_splits = a.max_folds if a.protocol == "cv" else 1
res = {(v, e): dict(raw=[], post=[], adj=[], pred=[], true=[], bat=[], pc=[]) for v in variants for e in evals}
for fi in range(n_splits):
    for name in variants:
        d = data[name]
        tr = (d["fold"] != fi) if a.protocol == "cv" else (d["split"] == "train")
        m = F.build_model().fit(d["X"][tr], d["y"][tr])
        for en, fr in evals.items():
            va = (fr.fold.values == fi) if a.protocol == "cv" else (fr.split.values == "validation")
            Pv = m.predict_proba(d["ev"][en][va])
            f2 = fr[va].reset_index(drop=True)
            y = f2.label.astype(int).values
            raw, post = F.LABELS[Pv.argmax(1)], F.postprocess(f2, Pv, a.tau)
            r = res[(name, en)]
            r["raw"].append(f1_score(y, raw, labels=L, average="macro", zero_division=0))
            r["post"].append(f1_score(y, post, labels=L, average="macro", zero_division=0))
            r["adj"].append(float((np.abs(post - y) <= 1).mean()))
            r["pred"].append(post); r["true"].append(y); r["bat"].append(f2.bat_class.values)
            r["pc"].append(f1_score(y, post, labels=L, average=None, zero_division=0))
    print(f"fold {fi}: " + " | ".join(f"{v}/{e} {res[(v, e)]['post'][-1]:.4f}" for v in variants for e in evals), flush=True)

rows = []
for (v, e), r in res.items():
    yt, yp, bt = np.concatenate(r["true"]), np.concatenate(r["pred"]), np.concatenate(r["bat"])
    pc = np.mean(r["pc"], axis=0)
    print(f"\n== {v} | 평가 {e} | aug={a.aug}: F1(후처리) {np.mean(r['post']):.4f} ± {np.std(r['post']):.4f} | 후처리 없음 {np.mean(r['raw']):.4f} | 인접 정확도 {np.mean(r['adj']):.4f}")
    print("   단계별 F1:", [round(float(x), 3) for x in pc], "| bat_class별 macro:",
          {b: round(f1_score(yt[bt == b], yp[bt == b], labels=sorted(set(yt[bt == b])), average="macro", zero_division=0), 3) for b in (1, 2, 3)})
    rows.append(dict(models="+".join(a.models) or "-", protocol=a.protocol, aug=a.aug, aug_all=a.aug_all, variant=v, eval=e,
                     f1_post=np.mean(r["post"]), f1_std=np.std(r["post"]), f1_raw=np.mean(r["raw"]), adj=np.mean(r["adj"]),
                     per_class=[round(float(x), 3) for x in pc]))
if a.models and n_splits > 1:
    for e in evals:
        dlt = np.array(res[("시계열+이미지", e)]["post"]) - np.array(res[("시계열만", e)]["post"])
        print(f"\n[{e}] fold별 차이(+이미지 - 시계열만): {np.round(dlt, 4)} 평균 {dlt.mean():+.4f} | 이긴 fold {int((dlt > 0).sum())}/{len(dlt)}")
elif a.models:
    for e in evals:
        print(f"\n[{e}] 차이(+이미지 - 시계열만): {res[('시계열+이미지', e)]['post'][0] - res[('시계열만', e)]['post'][0]:+.4f} (Val 19개 실험, ±0.02~0.03 변동 가능)")
p = ROOT / "outputs" / "logs" / "fusion_results.csv"
pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)
