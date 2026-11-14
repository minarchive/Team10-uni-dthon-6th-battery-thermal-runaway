"""부분 파인튜닝(실험 단위 5-fold CV). frozen probe 결과(중간 레이어가 최고)를 반영해 백본을 --cut 지점까지만 쓴다.

과적합 대응(실험 97개뿐):
- 백본을 잘라 파라미터 축소(--cut), layer-wise lr decay(--ld), weight decay, head dropout, 첫 epoch head만 학습(--head_warmup)
- epoch 당 전체 프레임의 --frac 만 샘플링(프레임이 5초 간격으로 거의 동일), 클래스 균형 샘플링(--power)
- 증강은 기하(소폭 zoom/shift/rot)+가우시안 노이즈만. 색(hue/밝기) 변환 없음(온도 의미 보존)
- 순서형 soft label(--tau): 인접 단계 오류에 관대한 목표 분포
- early stopping 은 train fold 안쪽 실험 단위 inner split(--mon_frac)의 Macro F1로만 결정, outer fold 는 점수 보고 전용
산출: outputs/logs/ft_results.csv, outputs/preds/ft_<name>_oof.npy (라벨 행, folds.parquet 순서, (N,6) 확률)
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from common import NUM_CLASSES, ROOT, table

TIMM = {
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "mae_b": "vit_base_patch16_224.mae",
    "siglip_b": "vit_base_patch16_siglip_224.webli",
    "resnet50": "resnet50.a1_in1k",
    "convnext_s": "convnext_small.fb_in22k_ft_in1k",
}
DEFAULT_CUT = {"dino_r50": 4, "resnet50": 3, "convnext_s": 3, "dinov2_s": 6, "dinov2_b": 8, "mae_b": 8, "siglip_b": 6}

ap = argparse.ArgumentParser()
ap.add_argument("--enc", required=True, choices=list(DEFAULT_CUT))
ap.add_argument("--name", default=None, help="실험 이름(기본: enc)")
ap.add_argument("--cut", type=int, default=0, help="ViT: 사용할 블록 수 / CNN: 사용할 stage 수 (0=기본값)")
ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--leaky", type=float, default=0.1)
ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--lr", type=float, default=1e-4, help="백본 최상위 단위 lr")
ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--ld", type=float, default=0.8, help="layer-wise lr decay")
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--head_warmup", type=int, default=1, help="처음 N epoch 은 head 만 학습")
ap.add_argument("--frac", type=float, default=0.3, help="epoch 당 샘플링 비율")
ap.add_argument("--power", type=float, default=0.5, help="클래스 균형 샘플링 지수 (0=없음, 1=완전 균형)")
ap.add_argument("--tau", type=float, default=0.5, help="순서형 soft label 온도 (0=one-hot+label smoothing)")
ap.add_argument("--ls", type=float, default=0.05, help="tau=0 일 때 label smoothing")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--aug", type=int, default=1)
ap.add_argument("--zoom", type=float, default=0.15)
ap.add_argument("--shift", type=float, default=0.08)
ap.add_argument("--rot", type=float, default=0.05, help="rad")
ap.add_argument("--noise", type=float, default=0.02)
ap.add_argument("--hflip", type=int, default=0)
ap.add_argument("--mon_frac", type=float, default=0.15)
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true", help="2000행·1 epoch·fold 0 동작 확인")
a = ap.parse_args()
name = a.name or a.enc
cut = a.cut or DEFAULT_CUT[a.enc]
dev = "cuda"
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
L = list(range(NUM_CLASSES))

df = table()
imgs = np.load(ROOT / "data" / "img224_pad.npy", mmap_mode="r")
assert len(imgs) == len(df)
y_all = df.label.values - 1
lab = (df.label.values >= 1) & (df.fold.values >= 0)
if a.smoke:
    a.epochs, a.folds, a.head_warmup = 1, "0", 0
    keep = np.sort(rng.choice(np.where(lab)[0], 2000, replace=False))
    lab = np.zeros(len(df), bool)
    lab[keep] = True


def mf1(yt, yp):
    return f1_score(yt, yp, average="macro", labels=L, zero_division=0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        if a.enc == "dino_r50":
            m = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            kw = {"img_size": 224} if "vit" in TIMM[a.enc] else {}
            m = timm.create_model(TIMM[a.enc], pretrained=True, **kw)
            mean, std = m.pretrained_cfg["mean"], m.pretrained_cfg["std"]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        if hasattr(m, "blocks"):
            self.kind = "vit"
            m.blocks = nn.Sequential(*list(m.blocks)[:cut])
            m.norm = nn.Identity()
            self.trunk, self.npf, dim = m, getattr(m, "num_prefix_tokens", 1), m.embed_dim
            self.post = nn.LayerNorm(dim)
        else:
            self.kind = "cnn"
            if hasattr(m, "stages"):
                seq = [m.stem] + list(m.stages)[:cut]
            else:
                act = m.act1 if hasattr(m, "act1") else m.relu
                seq = [m.conv1, m.bn1, act, m.maxpool] + [getattr(m, f"layer{i}") for i in range(1, cut + 1)]
            self.seq = nn.Sequential(*seq)
            self.seq.eval()
            with torch.no_grad():
                dim = self.seq(torch.zeros(1, 3, 224, 224)).shape[1]
            self.post = nn.Identity()
        d = a.drop
        self.head = (nn.Sequential(nn.Dropout(d), nn.Linear(dim, NUM_CLASSES)) if a.head == "linear" else
                     nn.Sequential(nn.Dropout(d), nn.Linear(dim, a.hidden), nn.LeakyReLU(a.leaky), nn.Dropout(d),
                                   nn.Linear(a.hidden, NUM_CLASSES)))
        self.dim = dim

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.kind == "vit":
            f = self.post(self.trunk.forward_features(x)[:, self.npf:].mean(1))
        else:
            f = self.seq(x).mean((2, 3))
        return self.head(f)

    def groups(self):
        if self.kind == "vit":
            blocks = list(self.trunk.blocks)
            misc = [p for n, p in self.trunk.named_parameters() if not n.startswith("blocks.")]
            units = [misc] + [list(b.parameters()) for b in blocks]
        else:
            units = [list(mm.parameters()) for mm in self.seq]
        units = [u for u in units if u]
        n = len(units)
        gs = [dict(params=u, base=a.lr * a.ld ** (n - 1 - i), backbone=True) for i, u in enumerate(units)]
        gs.append(dict(params=list(self.head.parameters()) + list(self.post.parameters()), base=a.head_lr, backbone=False))
        for g in gs:
            g["lr"] = g["base"]
        return gs


def augment(x):
    x = x.permute(0, 3, 1, 2).float() / 255
    if not a.aug:
        return x.contiguous(memory_format=torch.channels_last)
    B = x.shape[0]
    r = lambda: (torch.rand(B, device=x.device) * 2 - 1)
    s = 1 + r() * a.zoom
    ang = r() * a.rot
    flip = (torch.rand(B, device=x.device) < 0.5).float() * 2 - 1 if a.hflip else torch.ones(B, device=x.device)
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos * flip, -sin, r() * a.shift], 1), torch.stack([sin * flip, cos, r() * a.shift], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)
    if a.noise > 0:
        x = (x + torch.randn_like(x) * a.noise * torch.rand(B, 1, 1, 1, device=x.device)).clamp(0, 1)
    return x.contiguous(memory_format=torch.channels_last)


def batch_imgs(idx):
    o = np.argsort(idx)
    arr = np.empty((len(idx), 224, 224, 3), np.uint8)
    arr[o] = imgs[np.asarray(idx)[o]]
    return torch.from_numpy(arr).to(dev, non_blocking=True)


@torch.no_grad()
def predict(net, idx, bs=256):
    net.eval()
    out = []
    for i in range(0, len(idx), bs):
        x = batch_imgs(idx[i:i + bs]).permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(net(x.contiguous(memory_format=torch.channels_last)).float(), 1).cpu())
    return torch.cat(out).numpy()


cls_idx = torch.arange(NUM_CLASSES, device=dev)


def target(yb):
    if a.tau > 0:
        return torch.softmax(-(cls_idx[None] - yb[:, None]).abs().float() / a.tau, 1)
    t = F.one_hot(yb, NUM_CLASSES).float()
    return t * (1 - a.ls) + a.ls / NUM_CLASSES


def run_fold(f):
    va = np.where(lab & (df.fold.values == f))[0]
    tr = np.where(lab & (df.fold.values != f))[0]
    gi, gm = next(GroupShuffleSplit(1, test_size=a.mon_frac, random_state=a.seed).split(tr, groups=df.experiment_id.values[tr]))
    itr, imon = tr[gi], tr[gm]
    net = Net().to(dev).to(memory_format=torch.channels_last)
    gs = net.groups()
    opt = torch.optim.AdamW(gs, weight_decay=a.wd)
    cnt = np.bincount(y_all[itr], minlength=NUM_CLASSES)
    pw = (1.0 / np.maximum(cnt, 1)) ** a.power
    p = pw[y_all[itr]]
    p = p / p.sum()
    n_ep = max(1, int(a.frac * len(itr)))
    spe = math.ceil(n_ep / a.bs)
    total, warm = spe * a.epochs, max(1, spe // 2)
    best, best_state, bad, step = -1, None, 0, 0
    for ep in range(a.epochs):
        net.train()
        sel = itr[rng.choice(len(itr), n_ep, replace=True, p=p)]
        t0, tl = time.time(), 0.0
        for i in range(spe):
            b = sel[i * a.bs:(i + 1) * a.bs]
            if len(b) < 2:
                continue
            lr_s = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = g["base"] * lr_s * (0.0 if (g["backbone"] and ep < a.head_warmup) else 1.0)
            x = augment(batch_imgs(b))
            yb = torch.from_numpy(y_all[b]).long().to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
            loss = -(target(yb) * F.log_softmax(logits.float(), 1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            step += 1
        pm = predict(net, imon)
        fm = mf1(y_all[imon], pm.argmax(1))
        print(f"fold {f} ep {ep + 1}/{a.epochs} loss {tl / spe:.3f} mon_f1 {fm:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if fm > best:
            best, bad = fm, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif ep >= a.head_warmup:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(best_state)
    pv = predict(net, va)
    sub = rng.choice(itr, min(6000, len(itr)), replace=False)
    gap = mf1(y_all[sub], predict(net, sub).argmax(1)) - mf1(y_all[va], pv.argmax(1))
    return va, pv, gap, best


pos = np.cumsum(lab) - 1
oof = np.zeros((int(lab.sum()), NUM_CLASSES), np.float32)
fold_f1, gaps, covered = [], [], np.zeros(len(df), bool)
for f in [int(x) for x in a.folds.split(",")]:
    va, pv, gap, mon = run_fold(f)
    oof[pos[va]] = pv
    covered[va] = True
    fold_f1.append(mf1(y_all[va], pv.argmax(1)))
    gaps.append(gap)
    print(f"== fold {f}: outer F1 {fold_f1[-1]:.4f} (inner mon {mon:.3f}) gap {gap:.3f}", flush=True)

m = covered[lab]
yl, pl = y_all[lab][m], oof[m].argmax(1)
(ROOT / "outputs" / "preds").mkdir(parents=True, exist_ok=True)
np.save(ROOT / "outputs" / "preds" / f"ft_{name}_oof.npy", oof)
row = dict(name=name, enc=a.enc, cut=cut, folds=a.folds, f1_mean=float(np.mean(fold_f1)), f1_std=float(np.std(fold_f1)),
           oof_f1=mf1(yl, pl), adj_acc=float((np.abs(pl - yl) <= 1).mean()), stage_mae=float(np.abs(pl - yl).mean()),
           gap_train_minus_val=float(np.mean(gaps)),
           per_class_f1=json.dumps([round(v, 3) for v in f1_score(yl, pl, average=None, labels=L, zero_division=0)]),
           args=json.dumps({k: v for k, v in vars(a).items() if k not in ("name",)}))
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "args"}, flush=True)
if not a.smoke:
    out = ROOT / "outputs" / "logs" / "ft_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
