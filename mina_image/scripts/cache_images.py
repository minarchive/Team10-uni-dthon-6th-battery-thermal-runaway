"""파인튜닝용 전처리 이미지 캐시: 종횡비 유지 리사이즈(긴 변 224) + 정사각 패딩 -> data/img224_pad.npy (N,224,224,3) uint8.

행 순서 = data/folds.parquet. 패딩 색은 features/dinov2_s/meta.json의 pad_color(추출 때와 동일)를 쓰고,
없으면 처음 300장 테두리 중앙값으로 계산. 약 7GB, git에는 올리지 않는다(.gitignore).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from common import DATA, ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()

df = table()
paths = df.img_path.values
meta = ROOT / "features" / "dinov2_s" / "meta.json"
if meta.exists():
    PAD = tuple(json.load(open(meta))["pad_color"])
else:
    px = []
    for p in paths[:300]:
        x = np.asarray(Image.open(DATA / p).convert("RGB"))
        px.append(np.concatenate([x[0], x[-1], x[:, 0], x[:, -1]]))
    PAD = tuple(int(v) for v in np.median(np.concatenate(px), axis=0))
print("pad color", PAD)

S = a.size
out = np.lib.format.open_memmap(ROOT / "data" / f"img{S}_pad.npy", mode="w+", dtype=np.uint8, shape=(len(paths), S, S, 3))


def put(i):
    im = Image.open(DATA / paths[i]).convert("RGB")
    s = S / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
    c = Image.new("RGB", (S, S), PAD)
    c.paste(im, ((S - im.size[0]) // 2, (S - im.size[1]) // 2))
    out[i] = np.asarray(c)


with ThreadPoolExecutor(a.workers) as ex:
    for k, _ in enumerate(ex.map(put, range(len(paths)))):
        if k % 5000 == 0:
            print(k, "/", len(paths), flush=True)
out.flush()
print("saved", ROOT / "data" / f"img{S}_pad.npy", out.shape)
