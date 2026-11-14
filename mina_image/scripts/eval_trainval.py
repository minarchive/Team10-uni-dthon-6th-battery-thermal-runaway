"""공식 프로토콜(Train으로 학습 -> Validation으로 평가)로 상위 레이어 probe와 B0를 다시 점수화.

5-fold CV(Train+Val 합본) 결과가 Train->Val 단일 분할에서도 유지되는지 확인하는 용도.
레이어는 probe_results.csv의 5-fold 점수 상위 --topk개를 쓰므로 Val이 레이어 선택에 약하게 섞여 있다(보고 시 명시).
wd·epoch는 Train 안쪽 실험 단위 inner split에서만 선택. 산출: outputs/logs/trainval_results.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from common import NUM_CLASSES, ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--topk", type=int, default=3)
ap.add_argument("--wds", default="1e-4,1e-3,1e-2,1e-1")
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
wds = [float(w) for w in a.wds.split(",")]
L = list(range(NUM_CLASSES))
df = table()
y = df.label.values - 1
tr = np.where((df.split == "train").values)[0]
va = np.where((df.split == "validation").values)[0]
print("train rows", len(tr), "experiments", df.experiment_id.values[tr].shape and len(set(df.experiment_id.values[tr])),
      "| val rows", len(va), "experiments", len(set(df.experiment_id.values[va])))


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


res = pd.read_csv(ROOT / "outputs" / "logs" / "probe_results.csv")
res = res[(res.layer != "-") & ~res.encoder.str.contains("smoke")].drop_duplicates(["encoder", "layer"], keep="last")
sel = res.sort_values("f1_mean", ascending=False).groupby("encoder").head(a.topk)
rows = []

# B0 (같은 분할)
df["ratio"] = df.W / df.H
for name, cols in {"B0_size": ["W", "H", "ratio"], "B0c_cond": ["bat_class", "bat_cc", "soc_status", "tr_method"]}.items():
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, class_weight="balanced", random_state=0)
    clf.fit(df.loc[tr, cols], y[tr])
    p = clf.predict(df.loc[va, cols])
    rows.append(dict(encoder=name, layer="-", val_f1=mf1(y[va], p), cv_f1=np.nan, adj_acc=float((np.abs(p - y[va]) <= 1).mean()),
                     per_class_f1=json.dumps([round(v, 3) for v in f1_score(y[va], p, average=None, labels=L, zero_division=0)])))
    print(rows[-1], flush=True)

for _, r in sel.iterrows():
    fdir = ROOT / "features" / r.encoder
    idx = np.load(fdir / "idx.npy")
    assert len(idx) == len(df), "전체 행 feature만 지원"
    X = torch.from_numpy(np.load(fdir / f"{r.layer}.npy")).float().to(dev)
    Y = torch.from_numpy(np.where(y >= 0, y, 0)).long().to(dev)
    itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=a.seed).split(tr, groups=df.experiment_id.values[tr]))
    itr, iva = tr[itr], tr[iva]
    cand = [(*fit(X[itr], Y[itr], X[iva], Y[iva], wd), wd) for wd in wds]
    m, mu, sd, best, wd = max(cand, key=lambda c: c[3])
    with torch.no_grad():
        pv = m((X[va] - mu) / sd).argmax(1).cpu().numpy()
        ptr = m((X[itr] - mu) / sd).argmax(1).cpu().numpy()
    row = dict(encoder=r.encoder, layer=r.layer, val_f1=mf1(y[va], pv), cv_f1=float(r.f1_mean), wd=wd,
               adj_acc=float((np.abs(pv - y[va]) <= 1).mean()), train_minus_val=mf1(y[itr], ptr) - mf1(y[va], pv),
               per_class_f1=json.dumps([round(v, 3) for v in f1_score(y[va], pv, average=None, labels=L, zero_division=0)]))
    rows.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs" / "logs" / "trainval_results.csv", index=False)
print(out[["encoder", "layer", "val_f1", "cv_f1", "adj_acc"]].round(3).to_string(index=False))
