"""Train+Val 이미지를 라벨(1~6)별 폴더로 복사한다.

사용: python scripts/split_by_label.py --data <battery_data_light 경로> --out data_label_split
출력: <out>/level_{1..6}/<sample_id>.png, <out>/manifest.csv (sample_id,split,experiment_id,label,bat_class,tr_method,W,H)
test는 라벨이 없어 제외. 이미 있는 파일은 건너뛰므로 재실행 가능.
"""
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="data_label_split")
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()

data, out = Path(a.data), Path(a.out)
frames = []
for split in ["train", "validation"]:
    df = pd.read_csv(data / split / f"{split}.csv")
    df["split"] = split
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
for lv in range(1, 7):
    (out / f"level_{lv}").mkdir(parents=True, exist_ok=True)


def copy(row):
    src = data / row.split / row.image_path
    dst = out / f"level_{int(row.label)}" / f"{row.sample_id}.png"
    if not dst.exists():
        shutil.copy2(src, dst)


with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(copy, df.itertuples()))

df[["sample_id", "split", "experiment_id", "relative_time_s", "label", "bat_class", "bat_cc", "soc_status", "tr_method"]].to_csv(
    out / "manifest.csv", index=False
)
print(df.groupby(["label", "split"]).size().unstack(fill_value=0))
