"""frozen 인코더의 모든 블록/스테이지 출력을 풀링해 features/<tag>/<layer>.npy(fp16)로 캐시.

ViT: b01_cls, b01_mean, ... / CNN: s1_gap, s1_max, ...   (행 순서 = data/folds.parquet)
전처리: --policy pad(종횡비 유지+정사각 패딩, 기본) | distort(정사각 왜곡 리사이즈)
"""
import argparse
import json
import time

import numpy as np
import torch
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import DATA, ROOT, table

ENC = {
    "dinov2_s": dict(kind="timm", name="vit_small_patch14_dinov2.lvd142m"),
    "dinov2_b": dict(kind="timm", name="vit_base_patch14_dinov2.lvd142m"),
    "convnext_s": dict(kind="timm", name="convnext_small.fb_in22k_ft_in1k"),
    "resnet50": dict(kind="timm", name="resnet50.a1_in1k"),
    "siglip_b": dict(kind="timm", name="vit_base_patch16_siglip_224.webli"),
    "mae_b": dict(kind="timm", name="vit_base_patch16_224.mae"),
    "dino_r50": dict(kind="hub", name="dino_resnet50"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", required=True, choices=list(ENC))
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--policy", default="pad", choices=["pad", "distort"])
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="스모크 테스트용 무작위 N행")
ap.add_argument("--force", action="store_true", help="이미 추출된 태그도 다시 추출")
a = ap.parse_args()

df = table()
idx = np.arange(len(df))
tag = a.encoder + ("" if a.policy == "pad" else f"_{a.policy}") + (f"_s{a.size}" if a.size != 224 else "")
if a.limit:
    idx = np.sort(np.random.default_rng(0).choice(len(df), a.limit, replace=False))
    tag += "_smoke"
out = ROOT / "features" / tag
if (out / "meta.json").exists() and not a.force:
    print(f"skip {tag}: already extracted (use --force to redo)")
    raise SystemExit(0)
out.mkdir(parents=True, exist_ok=True)
np.save(out / "idx.npy", idx)
paths = df.img_path.values[idx]


def border_color(n=300):
    px = []
    for p in paths[:n]:
        x = np.asarray(Image.open(DATA / p).convert("RGB"))
        px.append(np.concatenate([x[0], x[-1], x[:, 0], x[:, -1]]))
    return tuple(int(v) for v in np.median(np.concatenate(px), axis=0))


PAD = border_color()


class DS(Dataset):
    def __len__(self):
        return len(paths)

    def __getitem__(self, i):
        im = Image.open(DATA / paths[i]).convert("RGB")
        S = a.size
        if a.policy == "distort":
            im = im.resize((S, S), Image.BICUBIC)
        else:
            s = S / max(im.size)
            im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
            c = Image.new("RGB", (S, S), PAD)
            c.paste(im, ((S - im.size[0]) // 2, (S - im.size[1]) // 2))
            im = c
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)


cfg = ENC[a.encoder]
dev = "cuda"
if cfg["kind"] == "timm":
    kw = {"img_size": a.size} if "vit" in cfg["name"] else {}
    model = timm.create_model(cfg["name"], pretrained=True, **kw)
    mean, std = model.pretrained_cfg["mean"], model.pretrained_cfg["std"]
else:
    model = torch.hub.load("facebookresearch/dino:main", cfg["name"])
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
model = model.eval().to(dev)
mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1) * 255
std = torch.tensor(std, device=dev).view(1, 3, 1, 1) * 255

if hasattr(model, "blocks"):
    mods = [(f"b{i + 1:02d}", m) for i, m in enumerate(model.blocks)]
elif hasattr(model, "stages"):
    mods = [(f"s{i + 1}", m) for i, m in enumerate(model.stages)]
else:
    mods = [(f"s{i}", getattr(model, f"layer{i}")) for i in range(1, 5)]

cur = {}


def mk(name):
    def hook(_, __, o):
        o = o[0] if isinstance(o, tuple) else o
        if o.ndim == 3:
            npf = getattr(model, "num_prefix_tokens", 1)
            if npf > 0:
                cur[name + "_cls"] = o[:, 0]
            cur[name + "_mean"] = o[:, npf:].mean(1)
        else:
            cur[name + "_gap"] = o.mean((2, 3))
            cur[name + "_max"] = o.amax((2, 3))
    return hook


for n, m in mods:
    m.register_forward_hook(mk(n))

store = {}
t0 = time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for bi, x in enumerate(DataLoader(DS(), batch_size=a.bs, num_workers=a.workers)):
        cur.clear()
        model((x.to(dev).float() - mean) / std) if cfg["kind"] == "hub" else model.forward_features((x.to(dev).float() - mean) / std)
        for k, v in cur.items():
            store.setdefault(k, []).append(v.float().cpu().half().numpy())
        if bi % 20 == 0:
            print(f"{tag} batch {bi} {time.time() - t0:.0f}s", flush=True)

for k, v in store.items():
    np.save(out / f"{k}.npy", np.concatenate(v))
json.dump(dict(encoder=a.encoder, timm_or_hub=cfg["name"], size=a.size, policy=a.policy, pad_color=PAD,
               mean=[float(m) for m in mean.flatten() / 255], std=[float(s) for s in std.flatten() / 255],
               layers=sorted(store), rows=int(len(idx)), seconds=round(time.time() - t0)),
          open(out / "meta.json", "w"), indent=1)
print("saved", out, sorted(store))
