"""인과적 HMM 필터로 센서 확률을 후처리: 확신 cummax 와 비교 (OOF 5-fold, 97개 실험).

모델: 은닉 단계 s∈{1..6}, 한 칸(5초)마다 머무르거나(1-h_s) 다음 단계로 진행(h_s). 라벨이 단조 증가·한 칸씩 진행한다는 구조를 쓴다.
관측: 센서 프레임 확률 P_t(s) 를 우도 (P_t(s)/prior_s)^beta 로 사용. 전방 필터(forward)만 쓰므로 현재·과거 프레임만 사용(미래 정보 없음).
간격이 5초가 아닌 프레임(Test 10/15초)은 k=round(dt/5) 칸 전이를 적용한다.
h_s 는 각 fold 의 학습 fold 실험에서만 추정한다(진행 횟수/단계 체류 프레임 수). gamma 로 h 를 스케일, beta 로 관측 가중을 조절해 격자 탐색(OOF 로만 선택).
"""
import argparse
import itertools

import numpy as np
import pandas as pd

import fuse_common as fc
from common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--ts_dir", default=str(ROOT / "ts_model"))
a = ap.parse_args()
F = fc.load_features_module(a.ts_dir)
c = fc.load(F, "ft_ord_dinov2_b", official_model="/nonexistent")
fr, y, P, fold = c.fr_lab, c.y_lab, c.sens_oof, c.fr_lab.fold.values
ex = fr.experiment_id.values
t = fr.relative_time_s.values.astype(float)
order = {e: np.where(ex == e)[0][np.argsort(t[ex == e], kind="stable")] for e in np.unique(ex)}


def est(train_exps):
    """학습 실험에서 단계별 진행 hazard 와 라벨 prior."""
    n_rows, n_adv = np.zeros(6), np.zeros(6)
    for e in train_exps:
        yy = y[order[e]] - 1
        for s in range(6):
            n_rows[s] += (yy[:-1] == s).sum()
            n_adv[s] += ((yy[:-1] == s) & (yy[1:] > s)).sum()
    h = n_adv / np.maximum(n_rows, 1)
    prior = np.bincount(np.concatenate([y[order[e]] - 1 for e in train_exps]), minlength=6) + 1.0
    return h, prior / prior.sum()


def filt(idx_exps, h, prior, beta, gamma):
    out = np.zeros(len(y), int)
    hh = np.clip(h * gamma, 1e-6, 0.5)
    hh[5] = 0.0
    for e in idx_exps:
        ix = order[e]
        al = np.zeros(6); al[0] = 1.0
        prev_t = t[ix[0]]
        for j, i in enumerate(ix):
            k = 1 if j == 0 else max(1, int(round((t[i] - prev_t) / 5.0)))
            for _ in range(k if j > 0 else 0):
                nxt = al * (1 - hh)
                nxt[1:] += al[:-1] * hh[:-1]
                al = nxt
            like = (np.clip(P[i], 1e-6, 1) / prior) ** beta
            al = al * like
            al /= al.sum()
            out[i] = al.argmax() + 1
            prev_t = t[i]
    return out


exps = np.unique(ex)
fold_of = pd.Series(fold, index=ex).groupby(level=0).first()
base = fc.post(c, fr, P, a.tau)
f1b = [fc.f1(y[fold == k], base[fold == k]) for k in range(5)]
print(f"기준: 센서 + 확신 cummax(τ={a.tau}) OOF F1 {fc.f1(y, base):.4f} | fold {np.round(f1b, 4).tolist()}")
raw = P.argmax(1) + 1
print(f"      센서 argmax(후처리 없음) OOF F1 {fc.f1(y, raw):.4f}")
print("\nHMM 전방 필터 격자 (beta=관측 가중, gamma=진행 hazard 배율):")
res = []
for beta, gamma in itertools.product([0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 2.0]):
    pred = np.zeros(len(y), int)
    for k in range(5):
        tr_e = [e for e in exps if fold_of[e] != k]
        te_e = [e for e in exps if fold_of[e] == k]
        h, prior = est(tr_e)
        pk = filt(te_e, h, prior, beta, gamma)
        pred[fold == k] = pk[fold == k]
    fs = [fc.f1(y[fold == k], pred[fold == k]) for k in range(5)]
    d = np.array(fs) - np.array(f1b)
    res.append((beta, gamma, fc.f1(y, pred), np.mean(fs), int((d > 0).sum()), pred))
    print(f"  beta={beta:<3} gamma={gamma:<3} OOF F1 {res[-1][2]:.4f} (기준 대비 {res[-1][2] - fc.f1(y, base):+.4f}) | 이긴 fold {res[-1][4]}/5 | 단계별 {fc.pcf1(y, pred)}", flush=True)
best = max(res, key=lambda r: r[2])
print(f"\n최고: beta={best[0]} gamma={best[1]} OOF F1 {best[2]:.4f} (기준 {fc.f1(y, base):.4f}, {best[2] - fc.f1(y, base):+.4f})")
print("※ 격자 12개 중 OOF 최고를 고른 것이라 약간 낙관적. 기준 대비 이득이 0.003 미만이면 잡음으로 본다.")
np.save(ROOT / "outputs" / "preds" / "hmm_best_oof_pred.npy", best[5])
