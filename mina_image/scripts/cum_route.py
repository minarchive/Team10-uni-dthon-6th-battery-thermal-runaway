"""실험 2: 누적 임계값 분해 + 임계값별 참조 모델.

라벨이 단조 증가하므로 6-class 를 5개 이진 문제 P(단계>=k+1), k=1..5 로 본다. 임계값마다 센서와 이미지의 logit 을
z_k = (1-w_k) logit(센서) + w_k logit(이미지) 로 섞고, w_k 는 Train 행 OOF 의 이진 logloss 가 최소인 값으로 고른다(임계값당 파라미터 1개).
가설: w_k > 0 은 '>=L3' 임계값(k=2)뿐이고 나머지는 0(센서만)이어야 한다. 결과가 이를 확인하거나 반박한다.
통제: 모든 임계값에 같은 w 를 쓰는 균일 블렌드. 단계 확률 변환은 finetune.py 의 순서형 변환과 같다(cummin 후 인접 차이).
평가: A) OOF 전체  B) 공식 프로토콜 Validation(cap500, full) — w 는 Train 행으로만 선택.
"""
import argparse

import numpy as np

import fuse_common as fc
from common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--img", default="ft_ord_dinov2_b")
ap.add_argument("--ws", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.8")
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--final", action="store_true")
ap.add_argument("--name", default="cum_route")
ap.add_argument("--n_jobs", type=int, default=6)
ap.add_argument("--ts_dir", default=str(ROOT / "ts_model"))
a = ap.parse_args()
F = fc.load_features_module(a.ts_dir)
c = fc.load(F, a.img, n_jobs=a.n_jobs)
ws = [float(x) for x in a.ws.split(",")]
eps = 1e-4
logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
sig = lambda z: 1 / (1 + np.exp(-z))
cum = lambda Pm: np.stack([Pm[:, k:].sum(1) for k in range(1, 6)], 1)  # 열 j = P(단계 >= j+2)
NAME = ["≥L2", "≥L3", "≥L4", "≥L5", "≥L6"]


def to6(S):
    S = np.minimum.accumulate(S, axis=1)
    p = (np.concatenate([np.ones((len(S), 1)), S], 1) - np.concatenate([S, np.zeros((len(S), 1))], 1)).clip(1e-6)
    return p / p.sum(1, keepdims=True)


def blend(Ps, Pi, w):
    w = np.broadcast_to(np.asarray(w, float), (5,))
    return to6(sig((1 - w) * logit(cum(Ps)) + w * logit(cum(Pi))))


def logloss(S, t):
    S = np.clip(S, eps, 1 - eps)
    return float(-(t * np.log(S) + (1 - t) * np.log(1 - S)).mean())


# ---- 임계값별 w 선택(Train 행 OOF logloss)
tr = c.trm
Cs, Ci = cum(c.sens_oof), cum(c.img_oof)
tgt = np.stack([(c.y_lab >= k + 2) for k in range(1, 6)], 1).astype(float) if False else np.stack([(c.y_lab >= j + 2).astype(float) for j in range(5)], 1)
print("\n임계값별 OOF(Train 행) 이진 logloss  (행=w, 열=임계값; 센서 w=0 이 기준)")
tab = np.zeros((len(ws), 5))
for i, w in enumerate(ws):
    z = sig((1 - w) * logit(Cs[tr]) + w * logit(Ci[tr]))
    tab[i] = [logloss(z[:, j], tgt[tr][:, j]) for j in range(5)]
print("w     " + "  ".join(f"{n:>8s}" for n in NAME))
for i, w in enumerate(ws):
    print(f"{w:<5.2f} " + "  ".join(f"{tab[i, j]:8.4f}" for j in range(5)))
wk = np.array([ws[int(tab[:, j].argmin())] for j in range(5)])
gain = tab[0] - tab.min(0)
print("\n선택된 w_k:", dict(zip(NAME, wk.tolist())), "| 센서 대비 logloss 감소:", dict(zip(NAME, np.round(gain, 4).tolist())))
wu = ws[int(np.argmin([np.mean([tab[i, j] for j in range(5)]) for i in range(len(ws))]))]
print("균일 블렌드(모든 임계값 같은 w) 최적 w:", wu)

# ---- A. OOF 전체
y = c.y_lab
cfgs = {"센서 단독": np.zeros(5), f"균일 w={wu}": np.full(5, wu), "임계값별 w_k": wk}
print("\n== A. OOF 전체 (w 는 Train 행으로 선택, 행 단위 5개 파라미터)")
for nm, w in cfgs.items():
    Pb = blend(c.sens_oof, c.img_oof, w)
    print(f"  {nm:14s} F1 {fc.f1(y, fc.post(c, c.fr_lab, Pb, a.tau)):.4f} | Train행 {fc.f1(y[tr], fc.post(c, c.fr_lab[tr], Pb[tr], a.tau)):.4f} | Val행 {fc.f1(y[c.vam], fc.post(c, c.fr_lab[c.vam], Pb[c.vam], a.tau)):.4f}")

# ---- B. 공식 프로토콜
print("\n== B. 공식 프로토콜 Validation (w 는 Train 행으로만 선택)")
for nm, fr, Ps, Pi, idx in fc.val_views(c):
    yv = fr.label.astype(int).values
    r = {k: fc.f1(yv, fc.post(c, fr, blend(Ps, Pi, w), a.tau)) for k, w in cfgs.items()}
    pc0, pc1 = fc.pcf1(yv, fc.post(c, fr, blend(Ps, Pi, cfgs["센서 단독"]), a.tau)), fc.pcf1(yv, fc.post(c, fr, blend(Ps, Pi, wk), a.tau))
    print(f"  [{nm}] " + " | ".join(f"{k} {v:.4f}" for k, v in r.items()) + f" | 임계값별-센서 {r['임계값별 w_k'] - r['센서 단독']:+.4f}")
    print(f"        단계별 센서 {pc0} -> 임계값별 {pc1}")

if a.final:
    fc.log("최종: 임계값별 w_k 로 Test 예측")
    views = fc.val_views(c)
    nm, fr, Ps, Pi, idx = views[0] if "cap500" in views[0][0] else views[1]
    yv = fr.label.astype(int).values
    vf = fc.f1(yv, fc.post(c, fr, blend(Ps, Pi, wk), a.tau))
    vn = int((fc.post(c, fr, blend(Ps, Pi, wk), a.tau) != fc.post(c, fr, blend(Ps, fc.neutral_img(c, len(fr)), wk), a.tau)).sum())
    lab_te = fc.post(c, c.fr_te, blend(c.sens_te, c.img_te, wk), a.tau)
    lab_n = fc.post(c, c.fr_te, blend(c.sens_te, fc.neutral_img(c, len(c.fr_te)), wk), a.tau)
    out, sub = fc.write_submission(c, lab_te, a.name)
    print("라벨 분포:", sub.label.value_counts(normalize=True).sort_index().round(3).to_dict(), "| w_k =", dict(zip(NAME, wk.tolist())))
    fc.record(a.name, vf, vn, int((lab_te != lab_n).sum()), out, f"views={nm}; w_k={wk.tolist()}")
