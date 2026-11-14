"""시계열 모델(features.py)에 이미지 정보를 여러 방식으로 결합했을 때를 같은 조건으로 비교 (+ 선택: 샘플링 간격 증강).

variants (--variants 쉼표 구분, 항상 ts 가 기준):
  ts      시계열만
  light   이미지 신호 2개(img_exp, img_p456)만 추가            = 이미지를 살짝만
  probs6  이미지 확률 6개 + img_exp 추가                        = 후기 결합(stacking)
  emb     프리즈 임베딩 PCA 성분 추가 (--emb enc:layer --emb_k 8) = early fusion
  (blend) --blend_ws 로 P=(1-w)P_ts+w P_img 가중치 스윕 (이미지 모델 확률 = --models 평균)
평가는 cap500(시계열 팀 프로토콜)과 mix(500행 초과 실험을 10/15초 혼합으로 다시 뽑음 = Test 긴 실험 구성) 두 가지, 확신 cummax 후처리 포함.
protocol: cv = data/folds.parquet 5-fold(Train+Val 97개 실험, 이미지 OOF 와 동일 분할) / trainval = Train 학습 -> Validation 평가
채택 기준(제안): ts 대비 평균 +0.01 이상, 5개 fold 중 4개 이상에서 승리, bat_class별 악화 없음.
주의: 이미지 확률(OOF)을 쓰는 변형은 2단계 평가라 약한 누수 가능성이 있다. emb 는 PCA 가 검증 fold 임베딩을 포함해 적합된다(라벨 미사용).
사용: python scripts/fusion_cv.py --ts_dir ts_model --models ft_dinov2_b --variants ts,light,probs6,emb --emb dino_r50:s4_gap
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import DATA, ROOT
from fusion_lib import IMG_VARIANTS, add_image_cols, emb_columns

ap = argparse.ArgumentParser()
ap.add_argument("--ts_dir", type=Path, required=True)
ap.add_argument("--models", nargs="+", default=[], help="image_probs.parquet 모델 태그(여러 개면 확률 평균)")
ap.add_argument("--probs", default="image_probs")
ap.add_argument("--variants", default="ts,light")
ap.add_argument("--emb", default="", help="enc:layer (예: dino_r50:s4_gap), emb 변형용")
ap.add_argument("--emb_k", type=int, default=8)
ap.add_argument("--blend_ws", default="0,0.1,0.2,0.3,0.5")
ap.add_argument("--protocol", default="cv", choices=["cv", "trainval"])
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--max_folds", type=int, default=5)
ap.add_argument("--aug", action="store_true")
ap.add_argument("--aug_all", action="store_true")
ap.add_argument("--aug_copies", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", type=Path, default=DATA)
a = ap.parse_args()
sys.path.insert(0, str(a.ts_dir))
import features as F  # 시계열 팀 코드

L = list(range(1, 7))
BASE0 = list(F.BASE_SIGNALS)
rng = np.random.default_rng(a.seed)
names = ["ts"] + [v for v in a.variants.split(",") if v and v != "ts"]
need_img = any(v in IMG_VARIANTS for v in names) or (a.models and a.blend_ws)
if any(v in IMG_VARIANTS for v in names):
    assert a.models, "light/probs6 는 --models 가 필요합니다"
if "emb" in names:
    assert a.emb, "emb 는 --emb enc:layer 가 필요합니다"

sens = pd.concat([pd.read_csv(a.data / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}).assign(split=s)
                  for s in ["train", "validation"]], ignore_index=True)
sens["sample_id"], sens["experiment_id"] = sens.sample_id.astype(str), sens.experiment_id.astype(str)
fp = ROOT / "data" / "folds.parquet"
if fp.exists():
    df = sens.merge(pd.read_parquet(fp)[["sample_id", "fold"]], on="sample_id", how="inner")
else:  # make_folds.py 와 같은 절차로 즉석 생성
    from sklearn.model_selection import StratifiedGroupKFold
    df = sens.copy()
    df["fold"] = -1
    for f_, (_, va_) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, df.label, df.experiment_id)):
        df.loc[df.index[va_], "fold"] = f_
assert len(df) == len(sens)
if a.models:
    df = add_image_cols(df, a.probs, a.models)
    assert df[["img_exp"]].notna().all().all(), "라벨 행에 이미지 OOF 확률이 없음"
if "emb" in names:
    df = pd.concat([df.reset_index(drop=True), emb_columns(df.sample_id.values, np.ones(len(df), bool), a.emb, a.emb_k, a.seed)], axis=1)
df = df.reset_index(drop=True)
EXTRA = {"ts": []}
for v in names:
    if v in IMG_VARIANTS:
        EXTRA[v] = IMG_VARIANTS[v]
    elif v == "emb":
        EXTRA[v] = [f"emb_{i}" for i in range(a.emb_k)]
print("rows", len(df), "experiments", df.experiment_id.nunique(), "| 변형:", names, "| aug:", a.aug, flush=True)


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


def resample(frame, pat, mode):
    """train_long: 긴(>500행) 실험만 다시 뽑고 짧은 실험 제외 / train_all: 전부 / eval: 긴 실험만 다시 뽑고 짧은 실험은 5초 그대로."""
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


def feats(frame, extra):
    F.BASE_SIGNALS = BASE0 + extra
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
for name in names:
    Xf = feats(df, EXTRA[name])
    cols = list(Xf.columns)
    Xs, ys, fs, ss = [Xf], [df.label.values], [df.fold.values], [df.split.values]
    for fr in aug_frames:
        Xs.append(feats(fr, EXTRA[name])[cols]); ys.append(fr.label.values); fs.append(fr.fold.values); ss.append(fr.split.values)
    data[name] = dict(X=pd.concat(Xs, ignore_index=True), y=np.concatenate(ys).astype(int), fold=np.concatenate(fs), split=np.concatenate(ss),
                      ev={en: feats(fr, EXTRA[name])[cols] for en, fr in evals.items()})
    print(f"{name}: feature {len(cols)}개, 학습 행 {len(data[name]['y'])}", flush=True)

n_splits = a.max_folds if a.protocol == "cv" else 1
res = {(v, e): dict(raw=[], post=[], adj=[], pred=[], true=[], bat=[], pc=[], P=[], fr=[]) for v in names for e in evals}
for fi in range(n_splits):
    for name in names:
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
            r["P"].append(Pv); r["fr"].append(f2)
    print(f"fold {fi}: " + " | ".join(f"{v}/{e} {res[(v, e)]['post'][-1]:.4f}" for v in names for e in evals), flush=True)

rows = []
for (v, e), r in res.items():
    yt, yp, bt = np.concatenate(r["true"]), np.concatenate(r["pred"]), np.concatenate(r["bat"])
    pc = np.mean(r["pc"], axis=0)
    print(f"\n== {v} | 평가 {e} | aug={a.aug}: F1(후처리) {np.mean(r['post']):.4f} ± {np.std(r['post']):.4f} | 후처리 없음 {np.mean(r['raw']):.4f} | 인접 정확도 {np.mean(r['adj']):.4f}")
    print("   단계별 F1:", [round(float(x), 3) for x in pc], "| bat_class별 macro:",
          {b: round(f1_score(yt[bt == b], yp[bt == b], labels=sorted(set(yt[bt == b])), average="macro", zero_division=0), 3) for b in (1, 2, 3)})
    rows.append(dict(models="+".join(a.models) or "-", emb=a.emb or "-", protocol=a.protocol, aug=a.aug, variant=v, eval=e,
                     f1_post=np.mean(r["post"]), f1_std=np.std(r["post"]), f1_raw=np.mean(r["raw"]), adj=np.mean(r["adj"]),
                     per_class=[round(float(x), 3) for x in pc]))

print("\n===== ts 대비 차이 (후처리 F1) =====")
for e in evals:
    base = np.array(res[("ts", e)]["post"])
    for v in names[1:]:
        dlt = np.array(res[(v, e)]["post"]) - base
        print(f"[{e}] {v:7s}: fold별 {np.round(dlt, 4)} | 평균 {dlt.mean():+.4f} | 이긴 fold {int((dlt > 0).sum())}/{len(dlt)}")

if a.models and a.blend_ws:
    print("\n===== 확률 가중 평균 blend: P=(1-w)P_ts + w P_img (이미지 모델 =", "+".join(a.models), ") =====")
    ic = [f"img_p{k}" for k in L]
    for e in evals:
        out = []
        for w in [float(x) for x in a.blend_ws.split(",")]:
            f1s = []
            for Pv, f2 in zip(res[("ts", e)]["P"], res[("ts", e)]["fr"]):
                pb = (1 - w) * Pv + w * f2[ic].values
                f1s.append(f1_score(f2.label.astype(int).values, F.postprocess(f2, pb, a.tau), labels=L, average="macro", zero_division=0))
            out.append((w, np.mean(f1s), np.std(f1s), f1s))
            rows.append(dict(models="+".join(a.models), emb="-", protocol=a.protocol, aug=a.aug, variant=f"blend_w{w}", eval=e,
                             f1_post=np.mean(f1s), f1_std=np.std(f1s), f1_raw=np.nan, adj=np.nan, per_class=[]))
        b0 = np.array(out[0][3])
        for w, m, s, f1s in out:
            d = np.array(f1s) - b0
            print(f"[{e}] w={w:<4}: F1 {m:.4f} ± {s:.4f} | w=0 대비 평균 {d.mean():+.4f}, 이긴 fold {int((d > 0).sum())}/{len(d)}")

p = ROOT / "outputs" / "logs" / "fusion_results.csv"
pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)
