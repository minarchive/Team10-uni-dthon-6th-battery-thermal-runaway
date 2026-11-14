"""라벨별 콜라주 + 간단한 색 통계 + 단계 전이 시간 EDA.

사용: python scripts/eda_label_collage.py --data <battery_data_light> --out outputs/figures
산출: collage_level_{1..6}.png (행=bat_class, 열=서로 다른 실험), overview.png(라벨×bat_class 평균상),
      stats_by_level.csv, stage_timing.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="outputs/figures")
ap.add_argument("--n", type=int, default=8, help="bat_class 행당 샘플 수")
ap.add_argument("--stat_n", type=int, default=150, help="라벨당 통계 표본 수")
a = ap.parse_args()
data, out = Path(a.data), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
T = 128

frames = []
for s in ["train", "validation"]:
    d = pd.read_csv(data / s / f"{s}.csv")
    d["split"] = s
    frames.append(d)
df = pd.concat(frames, ignore_index=True)


def load(r):
    return Image.open(data / r.split / r.image_path).convert("RGB")


def tile(im):
    s = T / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
    c = Image.new("RGB", (T, T), (255, 255, 255))
    c.paste(im, ((T - im.size[0]) // 2, (T - im.size[1]) // 2))
    return c


classes = sorted(df.bat_class.unique())
rng = np.random.default_rng(0)
mean_tiles = {}
for lv in range(1, 7):
    sub = df[df.label == lv]
    canvas = Image.new("RGB", (a.n * (T + 4) + 60, len(classes) * (T + 4) + 24), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 4), f"Level {lv}  (rows = bat_class {classes}, cols = different experiments)", fill=(0, 0, 0))
    for ri, bc in enumerate(classes):
        g = sub[sub.bat_class == bc]
        if g.empty:
            continue
        exps = g.experiment_id.unique()
        rng.shuffle(exps)
        picks = [g[g.experiment_id == e].sample(1, random_state=int(rng.integers(1e6))).iloc[0] for e in exps[: a.n]]
        arr = []
        for ci, r in enumerate(picks):
            t = tile(load(r))
            arr.append(np.asarray(t, dtype=float))
            canvas.paste(t, (60 + ci * (T + 4), 24 + ri * (T + 4)))
        dr.text((4, 24 + ri * (T + 4) + T // 2), f"class {bc}", fill=(0, 0, 0))
        mean_tiles[(lv, bc)] = Image.fromarray(np.mean(arr, 0).astype(np.uint8))
    canvas.save(out / f"collage_level_{lv}.png")

# 라벨 x bat_class 평균상
ov = Image.new("RGB", (6 * (T + 4) + 60, len(classes) * (T + 4) + 24), (255, 255, 255))
dr = ImageDraw.Draw(ov)
for lv in range(1, 7):
    dr.text((60 + (lv - 1) * (T + 4) + 40, 6), f"Level {lv}", fill=(0, 0, 0))
    for ri, bc in enumerate(classes):
        if (lv, bc) in mean_tiles:
            ov.paste(mean_tiles[(lv, bc)], (60 + (lv - 1) * (T + 4), 24 + ri * (T + 4)))
for ri, bc in enumerate(classes):
    dr.text((4, 24 + ri * (T + 4) + T // 2), f"class {bc}", fill=(0, 0, 0))
ov.save(out / "overview_mean_by_level.png")

# 색 통계(표본)
rows = []
for lv in range(1, 7):
    for _, r in df[df.label == lv].sample(a.stat_n, random_state=0).iterrows():
        x = np.asarray(load(r)).astype(int)
        R, G, B = x[..., 0], x[..., 1], x[..., 2]
        rows.append(dict(
            label=lv, bat_class=r.bat_class, W=x.shape[1], H=x.shape[0],
            magenta=((R > 150) & (B > 150) & (G < 90)).mean(),      # 저온 배경
            blue_cyan=((B > 150) & (R < 90)).mean(),
            green=((G > 150) & (R < 150) & (B < 150)).mean(),
            yellow=((R > 180) & (G > 180) & (B < 120)).mean(),
            red=((R > 180) & (G < 90) & (B < 90)).mean(),
            white=((R > 200) & (G > 200) & (B > 180)).mean(),         # 최고온
            hot=(((R > 180) & (G < 90) & (B < 90)) | ((R > 200) & (G > 200) & (B > 180))).mean(),
            uniq=len(np.unique(x.reshape(-1, 3), axis=0)),
        ))
st = pd.DataFrame(rows)
st.groupby("label").mean(numeric_only=True).round(3).to_csv(out / "stats_by_level.csv")
st.groupby(["label", "bat_class"]).mean(numeric_only=True).round(3).to_csv(out / "stats_by_level_class.csv")
print(st.groupby("label").mean(numeric_only=True).round(3).to_string())
print(st.groupby(["label", "bat_class"]).hot.agg(["mean", "size"]).round(3).unstack().to_string())

# 단계 전이 시간(실험별 첫 등장 시각, 마지막 등장 시각)
t = df.groupby(["experiment_id", "label"]).relative_time_s.agg(["min", "max", "size"]).reset_index()
t.to_csv(out / "stage_timing.csv", index=False)
print("stage 시작시각(초) 분포"); print(t.groupby("label")["min"].describe().round(0).to_string())
print("stage 지속 프레임 수"); print(t.groupby("label")["size"].describe().round(0).to_string())
print("실험당 stage 개수 / 라벨 단조성")
mono = df.sort_values(["experiment_id", "relative_time_s"]).groupby("experiment_id").label.apply(lambda s: bool((s.diff().dropna() >= 0).all()))
print("monotonic non-decreasing experiments:", mono.sum(), "/", len(mono))
print("label x bat_class counts"); print(df.groupby(["label", "bat_class"]).size().unstack(fill_value=0).to_string())
print("label x tr_method counts"); print(df.groupby(["label", "tr_method"]).size().unstack(fill_value=0).to_string())
