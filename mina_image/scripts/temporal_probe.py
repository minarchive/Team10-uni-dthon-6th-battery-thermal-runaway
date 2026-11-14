"""시간 문맥 probe: 캐시된 프레임 임베딩 + '과거 프레임만' 쓰는 시간 기반 feature, 5/10/15초/혼합 간격 증강.

배경: Train/Val 은 5초 고정, Test 는 5초 9개 실험 + 10·15초가 한 실험 안에서 섞인 2개 실험(전체의 29%).
설계:
- 창은 행 수가 아니라 relative_time_s 기준: Δt 를 반영한 EMA(tau), t-L 이전 가장 가까운 프레임과의 차이(lag)
- 임베딩은 train fold 통계로 표준화 + PCA(--dim) (EMA/차이는 선형이라 PCA 후 계산해도 동일)
- 학습은 5초(원본)+10초+15초+혼합{10,15} 로 다시 뽑은 시퀀스를 모두 사용(시작 위치 무작위), 평가는 패턴별로 따로 보고
- mode=single(현재 프레임만, 같은 증강) 대비 mode=temporal 의 이득을 같은 조건으로 비교
- 모든 변환은 과거 행만 사용(자동 self-test: 시퀀스를 잘라도 앞부분 feature 불변)
산출: outputs/logs/temporal_results.csv, outputs/preds/temporal_<enc>__<layer>_oof.npy (5초 패턴, 라벨 행 순서)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from common import NUM_CLASSES, ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--enc", required=True)
ap.add_argument("--layer", required=True)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--taus", default="30,120")
ap.add_argument("--lags", default="30,120")
ap.add_argument("--copies", type=int, default=2, help="10/15/혼합 패턴당 무작위 시작 위치 수")
ap.add_argument("--wds", default="1e-4,1e-3,1e-2,1e-1")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
taus = [float(x) for x in a.taus.split(",")]
lags = [float(x) for x in a.lags.split(",")]
wds = [float(x) for x in a.wds.split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
L = list(range(NUM_CLASSES))
PATTERNS = ["5", "10", "15", "mix"]


def temporal(Z, t):
    """과거 행만 사용하는 시간 기반 feature. Z (n,d) 시간순, t (n,) 초."""
    out = [Z]
    for tau in taus:
        E = np.empty_like(Z)
        E[0] = Z[0]
        for i in range(1, len(Z)):
            al = np.exp(-(t[i] - t[i - 1]) / tau)
            E[i] = al * E[i - 1] + (1 - al) * Z[i]
        out.append(E)
    for lag in lags:
        j = np.maximum(np.searchsorted(t, t - lag, side="right") - 1, 0)
        out.append(Z - Z[j])
    return np.concatenate(out, 1)


def selftest():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 8)).astype(np.float32)
    t = np.cumsum(rng.choice([5.0, 10.0, 15.0], 60))
    full = temporal(Z, t)
    for cut in (1, 7, 30, 59):
        assert np.allclose(full[:cut], temporal(Z[:cut], t[:cut]), atol=1e-5), "미래 정보 누수"
    print("self-test ok: 미래 행을 잘라도 앞부분 feature 동일", flush=True)


selftest()


def subsample(n, pat, rng):
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


df = table()
lab = (df.label.values >= 1) & (df.fold.values >= 0)
fdir = ROOT / "features" / a.enc
assert len(np.load(fdir / "idx.npy")) == len(df)
E = np.load(fdir / f"{a.layer}.npy")[lab].astype(np.float32)
y = df.label.values[lab] - 1
fold = df.fold.values[lab]
times = df.relative_time_s.values[lab].astype(float)
exp = df.experiment_id.values[lab]
exp_rows = {}
for e in np.unique(exp):
    r = np.where(exp == e)[0]
    exp_rows[e] = r[np.argsort(times[r], kind="stable")]


def build(Z, exps, pats, copies, rng, mode):
    X, Y, G, R, P = [], [], [], [], []
    for e in exps:
        rows = exp_rows[e]
        for pat in pats:
            for _ in range(1 if pat == "5" else copies):
                k = subsample(len(rows), pat, rng)
                r = rows[k]
                F = temporal(Z[r], times[r]) if mode == "temporal" else Z[r]
                X.append(F)
                Y.append(y[r])
                G.append(exp[r])
                R.append(r)
                P.append(np.full(len(r), pats.index(pat)))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G), np.concatenate(R), np.concatenate(P)


def mf1(yt, yp):
    return f1_score(yt, yp, average="macro", labels=L, zero_division=0)


def fit(Xtr, ytr, Xva, yva, wd):
    torch.manual_seed(a.seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    cnt = torch.bincount(ytr, minlength=NUM_CLASSES).float().clamp(min=1)
    m = nn.Linear(Xtr.shape[1], NUM_CLASSES).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=(len(ytr) / (NUM_CLASSES * cnt)).to(dev))
    best, state, bad = -1, None, 0
    for _ in range(a.epochs):
        perm = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(perm), 2048):
            b = perm[i:i + 2048]
            opt.zero_grad()
            lossf(m(Xtr[b]), ytr[b]).backward()
            opt.step()
        with torch.no_grad():
            f = mf1(yva.cpu().numpy(), m(Xva).argmax(1).cpu().numpy())
        if f > best:
            best, bad, state = f, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    m.load_state_dict(state)
    return m, mu, sd, best


rng = np.random.default_rng(a.seed)
res = {(m, p): dict(f1=[], pred=[], true=[]) for m in ("single", "temporal") for p in PATTERNS}
oof_t = np.zeros((len(y), NUM_CLASSES), np.float32)
for f in range(5):
    tr_e = np.unique(exp[fold != f])
    va_e = np.unique(exp[fold == f])
    tr_rows = np.where(fold != f)[0]
    mu, sd = E[tr_rows].mean(0), E[tr_rows].std(0) + 1e-6
    sub = rng.choice(tr_rows, min(20000, len(tr_rows)), replace=False)
    pca = PCA(a.dim, svd_solver="randomized", random_state=a.seed).fit((E[sub] - mu) / sd)
    Z = pca.transform((E - mu) / sd).astype(np.float32)
    for mode in ("single", "temporal"):
        Xtr, Ytr, Gtr, _, _ = build(Z, tr_e, PATTERNS, a.copies, rng, mode)
        itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=a.seed).split(Xtr, groups=Gtr))
        Xt, Yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).long().to(dev)
        cand = [(*fit(Xt[itr], Yt[itr], Xt[iva], Yt[iva], wd), wd) for wd in wds]
        m, mu_f, sd_f, _, wd = max(cand, key=lambda c: c[3])
        for pi, pat in enumerate(PATTERNS):
            Xv, Yv, _, Rv, _ = build(Z, va_e, [pat], a.copies, np.random.default_rng(1000 + f), mode)
            with torch.no_grad():
                pv = torch.softmax(m((torch.from_numpy(Xv).to(dev) - mu_f) / sd_f), 1).cpu().numpy()
            r = res[(mode, pat)]
            r["f1"].append(mf1(Yv, pv.argmax(1)))
            r["pred"].append(pv.argmax(1))
            r["true"].append(Yv)
            if mode == "temporal" and pat == "5":
                oof_t[Rv] = pv
        print(f"fold {f} {mode} wd {wd}: " + " ".join(f"{p}s={res[(mode, p)]['f1'][-1]:.3f}" for p in PATTERNS), flush=True)

rows = []
for (mode, pat), r in res.items():
    yt, yp = np.concatenate(r["true"]), np.concatenate(r["pred"])
    rows.append(dict(enc=a.enc, layer=a.layer, mode=mode, pattern=pat if pat == "mix" else pat + "s", dim=a.dim,
                     f1_mean=float(np.mean(r["f1"])), f1_std=float(np.std(r["f1"])), pooled_f1=mf1(yt, yp),
                     adj_acc=float((np.abs(yp - yt) <= 1).mean()), stage_mae=float(np.abs(yp - yt).mean()),
                     args=json.dumps(dict(taus=taus, lags=lags, copies=a.copies))))
out = pd.DataFrame(rows)
print(out.drop(columns="args").round(4).to_string(index=False))
(ROOT / "outputs" / "preds").mkdir(parents=True, exist_ok=True)
np.save(ROOT / "outputs" / "preds" / f"temporal_{a.enc}__{a.layer}_oof.npy", oof_t)
p = ROOT / "outputs" / "logs" / "temporal_results.csv"
out.to_csv(p, mode="a", header=not p.exists(), index=False)
