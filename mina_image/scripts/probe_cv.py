"""캐시된 feature에 대한 GPU 선형 probe, 실험 단위 5-fold(folds.parquet) Macro F1.

과적합 대응:
- 표준화 통계·weight decay·early-stopping epoch는 전부 train fold 안쪽(실험 단위 inner split 20%)에서만 선택
- 바깥 fold(val)는 점수 보고에만 사용
- inner train F1 - outer val F1 = gap 을 같이 기록
산출: outputs/preds/<tag>__<layer>_oof.npy (라벨 있는 행, folds.parquet 순서, (N,6) 확률),
      outputs/logs/probe_results.csv (append)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from common import NUM_CLASSES, ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", required=True, help="features/ 아래 태그")
ap.add_argument("--layers", default="all", help="all 또는 쉼표 구분 목록(b12_cls,s4_gap ...)")
ap.add_argument("--wds", default="1e-4,1e-3,1e-2,1e-1")
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
fdir = ROOT / "features" / a.encoder
idx = np.load(fdir / "idx.npy")
df = table().iloc[idx].reset_index(drop=True)
lab = (df.label.values >= 1) & (df.fold.values >= 0)
y_all = df.label.values - 1
layers = sorted(p.stem for p in fdir.glob("*.npy") if p.stem != "idx") if a.layers == "all" else a.layers.split(",")
wds = [float(w) for w in a.wds.split(",")]
(ROOT / "outputs" / "preds").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)


def mf1(y, p):
    return f1_score(y, p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)


def fit(Xtr, ytr, Xva, yva, wd):
    torch.manual_seed(a.seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    cnt = torch.bincount(ytr, minlength=NUM_CLASSES).float().clamp(min=1)
    w = (len(ytr) / (NUM_CLASSES * cnt)).to(dev)
    m = nn.Linear(Xtr.shape[1], NUM_CLASSES).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=w)
    best, best_state, bad = -1, None, 0
    for ep in range(a.epochs):
        perm = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(perm), 2048):
            b = perm[i:i + 2048]
            opt.zero_grad()
            lossf(m(Xtr[b]), ytr[b]).backward()
            opt.step()
        with torch.no_grad():
            f = mf1(yva.cpu().numpy(), m(Xva).argmax(1).cpu().numpy())
        if f > best:
            best, bad = f, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    m.load_state_dict(best_state)
    return m, mu, sd, best


for layer in layers:
    X = torch.from_numpy(np.load(fdir / f"{layer}.npy")).float().to(dev)
    Y = torch.from_numpy(np.where(lab, y_all, 0)).long().to(dev)
    oof = np.zeros((int(lab.sum()), NUM_CLASSES), dtype=np.float32)
    pos = np.cumsum(lab) - 1  # df 행 -> oof 행
    fold_f1, gaps, chosen = [], [], []
    for f in range(5):
        va = np.where(lab & (df.fold.values == f))[0]
        tr = np.where(lab & (df.fold.values != f))[0]
        if len(va) == 0:
            continue
        gss = GroupShuffleSplit(1, test_size=0.2, random_state=a.seed)
        itr, iva = next(gss.split(tr, groups=df.experiment_id.values[tr]))
        itr, iva = tr[itr], tr[iva]
        cand = []
        for wd in wds:
            m, mu, sd, best = fit(X[itr], Y[itr], X[iva], Y[iva], wd)
            cand.append((best, wd, m, mu, sd))
        best, wd, m, mu, sd = max(cand, key=lambda c: c[0])
        with torch.no_grad():
            pv = torch.softmax(m((X[va] - mu) / sd), 1).cpu().numpy()
            ptr = m((X[itr] - mu) / sd).argmax(1).cpu().numpy()
        oof[pos[va]] = pv
        fold_f1.append(mf1(y_all[va], pv.argmax(1)))
        gaps.append(mf1(y_all[itr], ptr) - fold_f1[-1])
        chosen.append(wd)
    yl, pl = y_all[lab], oof.argmax(1)
    np.save(ROOT / "outputs" / "preds" / f"{a.encoder}__{layer}_oof.npy", oof)
    row = dict(encoder=a.encoder, layer=layer, dim=int(X.shape[1]), f1_mean=np.mean(fold_f1), f1_std=np.std(fold_f1),
               oof_f1=mf1(yl, pl), adj_acc=float((np.abs(pl - yl) <= 1).mean()), stage_mae=float(np.abs(pl - yl).mean()),
               gap_train_minus_val=float(np.mean(gaps)), wds=json.dumps(chosen),
               per_class_f1=json.dumps([round(v, 3) for v in f1_score(yl, pl, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)]))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = ROOT / "outputs" / "logs" / "probe_results.csv"
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
