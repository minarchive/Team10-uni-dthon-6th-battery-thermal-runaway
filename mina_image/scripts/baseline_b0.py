"""B0: 이미지 크기 메타만으로 얼마나 맞히는가(지름길 측정). B0c는 실험 조건(bat_class 등)만 사용. 동일 5-fold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from common import ROOT, table

df = table()
df = df[(df.label >= 1) & (df.fold >= 0)].reset_index(drop=True)
df["ratio"] = df.W / df.H
sets = {"B0_size": ["W", "H", "ratio"], "B0c_cond": ["bat_class", "bat_cc", "soc_status", "tr_method"],
        "B0sc_size+cond": ["W", "H", "ratio", "bat_class", "bat_cc", "soc_status", "tr_method"]}
y = df.label.values - 1
rows = []
for name, cols in sets.items():
    oof = np.zeros(len(df), int)
    fs = []
    for f in range(5):
        va, tr = df.fold.values == f, df.fold.values != f
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, class_weight="balanced", random_state=0)
        clf.fit(df.loc[tr, cols], y[tr])
        oof[va] = clf.predict(df.loc[va, cols])
        fs.append(f1_score(y[va], oof[va], average="macro", labels=list(range(6)), zero_division=0))
    r = dict(encoder=name, layer="-", dim=len(cols), f1_mean=np.mean(fs), f1_std=np.std(fs),
             oof_f1=f1_score(y, oof, average="macro", labels=list(range(6)), zero_division=0),
             adj_acc=float((np.abs(oof - y) <= 1).mean()), stage_mae=float(np.abs(oof - y).mean()),
             gap_train_minus_val=np.nan, wds="", per_class_f1="")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    rows.append(r)
out = ROOT / "outputs" / "logs" / "probe_results.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
