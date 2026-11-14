"""Train+Val 97개 실험을 실험 단위 StratifiedGroupKFold(5)로 분할하고 test를 붙여 data/folds.parquet 저장."""
import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from common import DATA, FOLDS

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

frames = []
for split in ["train", "validation", "test"]:
    d = pd.read_csv(DATA / split / f"{split}.csv")
    d["split"] = split
    if "label" not in d:
        d["label"] = 0
    d["img_path"] = split + "/" + d["image_path"]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["label"] = df["label"].astype(int)
df.loc[df.split == "test", "label"] = -1


def size(p):
    with Image.open(DATA / p) as im:
        return im.size


with ThreadPoolExecutor(32) as ex:
    sz = list(ex.map(size, df.img_path))
df["W"] = [s[0] for s in sz]
df["H"] = [s[1] for s in sz]

df["fold"] = -1
L = df[df.split != "test"]
sgkf = StratifiedGroupKFold(n_splits=a.k, shuffle=True, random_state=a.seed)
for f, (_, va) in enumerate(sgkf.split(L, L.label, L.experiment_id)):
    df.loc[L.index[va], "fold"] = f
assert df[df.split != "test"].groupby("experiment_id").fold.nunique().max() == 1, "실험이 fold에 걸쳐 나뉨"

keep = ["sample_id", "experiment_id", "relative_time_s", "split", "label", "fold", "bat_class", "bat_cc",
        "soc_status", "tr_method", "W", "H", "img_path"]
FOLDS.parent.mkdir(parents=True, exist_ok=True)
df[keep].to_parquet(FOLDS, index=False)

L = df[df.split != "test"]
print("rows", len(df), "labeled", len(L), "experiments", L.experiment_id.nunique())
print("experiments per fold:\n", L.groupby("fold").experiment_id.nunique().to_string())
print("label share per fold:\n", pd.crosstab(L.fold, L.label, normalize="index").round(3).to_string())
print("bat_class share per fold:\n", pd.crosstab(L.fold, L.bat_class, normalize="index").round(3).to_string())
print("saved", FOLDS)
