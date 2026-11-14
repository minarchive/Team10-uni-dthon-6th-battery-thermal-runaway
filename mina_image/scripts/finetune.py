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
    "effnet_b0": "efficientnet_b0.ra_in1k",
    "effnet_b3": "efficientnet_b3.ra2_in1k",
}
DEFAULT_CUT = {"effnet_b0": 6, "effnet_b3": 6, "dino_r50": 4, "resnet50": 3, "convnext_s": 3, "dinov2_s": 6, "dinov2_b": 8, "mae_b": 8, "siglip_b": 6}

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
ap.add_argument("--ordinal", type=int, default=0, help="1=순서형 누적 헤드(5개 sigmoid: P(단계>k)), 추론 시 6-class 확률로 변환")
ap.add_argument("--thr_w", default="1,2,2,1,1", help="누적 임계 k=0..4(>=L2,>=L3,>=L4,>=L5,>=L6) BCE 가중치. 센서가 지연되는 L2->L3, L3->L4 경계에 큰 값")
ap.add_argument("--bnd_gain", type=float, default=0.0, help="전환 경계 주변 프레임 샘플링 가중 추가량(0=끔). 가중=1+gain")
ap.add_argument("--bnd_rows", type=int, default=12, help="경계 전후 몇 행(5초 간격, 12행=60초)까지 가중")
ap.add_argument("--bnd_stages", default="3,4", help="해당 단계가 시작하는 전환만 경계로 취급")
ap.add_argument("--sens_gain", type=float, default=0.0, help="센서 OOF 가 틀리거나 확신 낮은 프레임의 샘플링 가중 추가량(0=끔). 가중=1+gain. outputs/preds/sensor_lgb_oof.npy 필요")
ap.add_argument("--sens_conf", type=float, default=0.7, help="센서 확신도가 이 값 미만이면 '약점 프레임'으로 취급")
ap.add_argument("--official", type=int, default=0, help="1=5-fold 대신 공식 분할(Train 학습->Validation 평가) 1회. 스크리닝용(빠름). OOF 파일은 Validation 행만 유효")
ap.add_argument("--prev_sec", type=float, default=0.0, help=">0 이면 같은 실험의 prev_sec 초 이전(없으면 첫 프레임) 프레임도 입력: 특징 [f, f-f_prev]. 과거 프레임만 사용(인과적)")
ap.add_argument("--imgs", default="img224_pad.npy", help="data/ 아래 캐시 파일(img224_pad.npy=RGB, img224_tidx.npy=온도 인덱스 3채널 복제)")
ap.add_argument("--stack", type=int, default=1, help=">1 이면 현재+과거 (stack-1)개 프레임을 채널로 쌓는다(effnet 전용). 과거 프레임만 사용(인과적)")
ap.add_argument("--stack_sec", type=float, default=30.0, help="스택된 프레임 간 간격(초)")
ap.add_argument("--bin23", type=int, default=0, help="1=L2/L3 프레임만으로 학습·조기종료(L2 대 L3 전용 모델). 예측은 모든 프레임에 대해 저장. 센서가 L2/L3 로 본 프레임에서 게이트로 쓰기 위함")
ap.add_argument("--sens_in", type=int, default=0, help="1=센서 OOF/test 확률(log)을 헤드 입력에 이어 붙여 종단간 학습(센서가 아는 정보는 주어지고 백본은 보완 정보만 학습). linear 헤드 전용, 센서 부분은 항등으로 초기화")
ap.add_argument("--diff3", type=int, default=0, help="1=입력을 [현재-첫 프레임, 현재-15초 전, 현재-60초 전] 3채널(1채널 차이, 128 중심)로 구성. 과거·첫 프레임만 사용(인과)")
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
NOUT = NUM_CLASSES - 1 if a.ordinal else NUM_CLASSES
THR_W = torch.tensor([float(x) for x in a.thr_w.split(",")], device=dev)


def to_prob(logits):
    """순서형 누적 로짓(B,5) -> 6-class 확률. 단조(cummin)로 보정 후 인접 차이를 확률로."""
    if not a.ordinal:
        return torch.softmax(logits, 1)
    s = torch.sigmoid(logits)
    s = torch.cummin(s, 1).values
    one, zero = torch.ones_like(s[:, :1]), torch.zeros_like(s[:, :1])
    p = (torch.cat([one, s], 1) - torch.cat([s, zero], 1)).clamp_min(1e-6)
    return p / p.sum(1, keepdim=True)

df = table()
imgs = np.load(ROOT / "data" / a.imgs, mmap_mode="r")
assert len(imgs) == len(df)
y_all = df.label.values - 1
def prev_for(sec):
    """같은 실험에서 t-sec 이하의 마지막 프레임(없으면 실험의 첫 프레임). 미래 프레임은 쓰지 않는다."""
    out = np.arange(len(df))
    if sec <= 0:
        return out
    _ex, _ts = df.experiment_id.astype(str).values, df.relative_time_s.values.astype(float)
    for _e in np.unique(_ex):
        _ix = np.where(_ex == _e)[0]
        _o = _ix[np.argsort(_ts[_ix], kind="stable")]
        _t = _ts[_o]
        _k = np.searchsorted(_t, _t - sec, side="right") - 1
        out[_o] = _o[np.where(_k >= 0, _k, 0)]
    return out


SENS = None
if a.sens_in:
    _so = np.load(ROOT / "outputs" / "preds" / "sensor_lgb_oof.npy"); _st = np.load(ROOT / "outputs" / "preds" / "sensor_lgb_test.npy")
    _full = np.zeros((len(df), 6), np.float32)
    _full[np.where((df.label.values >= 1) & (df.fold.values >= 0))[0]] = _so
    _full[np.where(df.label.values == -1)[0]] = _st
    SENS = torch.from_numpy(np.log(np.clip(_full, 1e-4, 1)) / 5.0).float().to(dev)
    assert a.head == "linear" and a.ordinal == 0, "--sens_in 은 linear 헤드, 6-class 전용"
prev_idx = prev_for(a.prev_sec)
stack_idx = [prev_for(k * a.stack_sec) for k in range(a.stack)] if a.stack > 1 else None
DIFF_IDX = [prev_for(1e9), prev_for(15.0), prev_for(60.0)] if a.diff3 else None
if a.stack > 1:
    assert a.enc.startswith("effnet"), "--stack 은 effnet 전용"
lab = (df.label.values >= 1) & (df.fold.values >= 0)
te_idx = np.where(df.label.values == -1)[0]  # test 프레임(라벨 없음): fold 모델들의 평균 확률을 저장
if a.official:
    a.folds = "0"
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
        rep_ = a.stack if a.stack > 1 else 1
        self.register_buffer("mean", torch.tensor(mean).repeat(rep_).view(1, 3 * rep_, 1, 1))
        self.register_buffer("std", torch.tensor(std).repeat(rep_).view(1, 3 * rep_, 1, 1))
        if a.enc.startswith("effnet"):  # timm EfficientNet: stem + blocks[:cut] (blocks 는 단계 목록이라 ViT 분기와 구분)
            self.kind = "cnn"
            if a.stack > 1:  # 첫 합성곱을 3K 채널로 확장(사전학습 가중치를 K번 복제 후 /K)
                c0 = m.conv_stem
                nc = nn.Conv2d(3 * a.stack, c0.out_channels, c0.kernel_size, c0.stride, c0.padding, bias=False)
                nc.weight.data = torch.cat([c0.weight.data] * a.stack, 1) / a.stack
                m.conv_stem = nc
            self.seq = nn.Sequential(m.conv_stem, m.bn1, *list(m.blocks)[:cut])
            self.seq.eval()
            with torch.no_grad():
                dim = self.seq(torch.zeros(1, 3 * rep_, 224, 224)).shape[1]
            self.post = nn.Identity()
        elif hasattr(m, "blocks"):
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
        hin = dim * (2 if a.prev_sec > 0 else 1) + (6 if a.sens_in else 0)
        self.head = (nn.Sequential(nn.Dropout(d), nn.Linear(hin, NOUT)) if a.head == "linear" else
                     nn.Sequential(nn.Dropout(d), nn.Linear(hin, a.hidden), nn.LeakyReLU(a.leaky), nn.Dropout(d),
                                   nn.Linear(a.hidden, NOUT)))
        if a.sens_in:  # 이미지 특징 dropout 은 forward 에서 따로 적용, 센서 부분은 항등으로 시작
            lin = nn.Linear(hin, NOUT)
            nn.init.normal_(lin.weight, std=0.01); nn.init.zeros_(lin.bias)
            with torch.no_grad(): lin.weight[:, -6:] = torch.eye(6) * 5.0   # log-prob/5 -> 5 배로 되돌려 센서 로짓 그대로 시작
            self.head = lin
        self.dim = dim

    def encode(self, x):
        x = (x - self.mean) / self.std
        if self.kind == "vit":
            return self.post(self.trunk.forward_features(x)[:, self.npf:].mean(1))
        return self.seq(x).mean((2, 3))

    def forward(self, x, sx=None):
        if a.sens_in:
            f = nn.functional.dropout(self.encode(x), a.drop, self.training)
            return self.head(torch.cat([f, sx], 1))
        if a.prev_sec > 0:  # x: (B,6,H,W) = [현재 3ch, 과거 3ch]
            f, fp = self.encode(x[:, :3]), self.encode(x[:, 3:])
            return self.head(torch.cat([f, f - fp], 1))
        return self.head(self.encode(x))

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


def _load(idx):
    idx = np.asarray(idx)
    o = np.argsort(idx)
    arr = np.empty((len(idx), 224, 224, 3), np.uint8)
    arr[o] = imgs[idx[o]]
    return torch.from_numpy(arr).to(dev, non_blocking=True)


def batch_imgs(idx):
    if a.diff3:
        idx = np.asarray(idx); cur = _load(idx)[..., :1].float(); return torch.cat([((cur - _load(ri[idx])[..., :1].float() + 255.0) / 2.0).clamp(0, 255) for ri in DIFF_IDX], -1).to(torch.uint8)
    if a.stack > 1:
        return torch.cat([_load(si[np.asarray(idx)]) for si in stack_idx], -1)
    cur = _load(idx)
    return torch.cat([cur, _load(prev_idx[np.asarray(idx)])], -1) if a.prev_sec > 0 else cur


@torch.no_grad()
def predict(net, idx, bs=256):
    net.eval()
    out = []
    for i in range(0, len(idx), bs):
        x = batch_imgs(idx[i:i + bs]).permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sx = SENS[torch.as_tensor(np.asarray(idx[i:i + bs]), device=dev)] if a.sens_in else None
            out.append(to_prob(net(x.contiguous(memory_format=torch.channels_last), sx).float()).cpu())
    return torch.cat(out).numpy()


cls_idx = torch.arange(NUM_CLASSES, device=dev)


def target(yb):
    if a.tau > 0:
        return torch.softmax(-(cls_idx[None] - yb[:, None]).abs().float() / a.tau, 1)
    t = F.one_hot(yb, NUM_CLASSES).float()
    return t * (1 - a.ls) + a.ls / NUM_CLASSES


def boundary_w(idx):
    """idx(df 행 번호) 각각의 샘플링 가중. 지정 단계가 시작하는 전환 지점 전후 --bnd_rows 행이면 1+bnd_gain, 아니면 1."""
    w = np.ones(len(idx))
    if a.bnd_gain <= 0:
        return w
    stages = {int(x) for x in a.bnd_stages.split(",")}
    sub = pd.DataFrame({"pos": np.arange(len(idx)), "e": df.experiment_id.values[idx], "t": df.relative_time_s.values[idx], "y": y_all[idx] + 1})
    for _, g in sub.sort_values(["e", "t"], kind="stable").groupby("e", sort=False):
        yy, pp = g.y.values, g.pos.values
        ch = np.where(np.diff(yy) != 0)[0] + 1
        ch = [c for c in ch if yy[c] in stages]
        for c in ch:
            lo, hi = max(0, c - a.bnd_rows), min(len(yy), c + a.bnd_rows)
            w[pp[lo:hi]] = 1 + a.bnd_gain
    return w


_SENS = {}


def sens_w(idx):
    """센서 OOF 가 틀리거나(argmax != y) 확신도 < --sens_conf 인 프레임을 1+sens_gain 배 가중.
    주의: 센서 OOF 는 다른 fold 모델이 만든 것이라 현재 검증 fold 의 라벨 정보가 가중치에 약하게 섞인다(피처가 아니라 가중치뿐)."""
    if a.sens_gain <= 0:
        return np.ones(len(idx))
    if "w" not in _SENS:
        so = np.load(ROOT / "outputs" / "preds" / "sensor_lgb_oof.npy")
        li = np.where((df.label.values >= 1) & (df.fold.values >= 0))[0]
        bad = np.zeros(len(df), bool)
        bad[li] = (so.argmax(1) != y_all[li]) | (so.max(1) < a.sens_conf)
        _SENS["w"] = 1 + a.sens_gain * bad
    return _SENS["w"][idx]


def ord_loss(logits, yb):
    tgt = (yb[:, None] > torch.arange(NOUT, device=dev)[None]).float()
    return (F.binary_cross_entropy_with_logits(logits, tgt, reduction="none") * THR_W).sum(1).mean() / THR_W.mean()


def run_fold(f):
    if a.official:
        va = np.where(lab & (df.split.values == "validation"))[0]
        tr = np.where(lab & (df.split.values == "train"))[0]
    else:
        va = np.where(lab & (df.fold.values == f))[0]
        tr = np.where(lab & (df.fold.values != f))[0]
    gi, gm = next(GroupShuffleSplit(1, test_size=a.mon_frac, random_state=a.seed).split(tr, groups=df.experiment_id.values[tr]))
    itr, imon = tr[gi], tr[gm]
    if a.bin23:
        itr, imon = itr[np.isin(y_all[itr], [1, 2])], imon[np.isin(y_all[imon], [1, 2])]
    net = Net().to(dev).to(memory_format=torch.channels_last)
    gs = net.groups()
    opt = torch.optim.AdamW(gs, weight_decay=a.wd)
    cnt = np.bincount(y_all[itr], minlength=NUM_CLASSES)
    pw = (1.0 / np.maximum(cnt, 1)) ** a.power
    p = pw[y_all[itr]] * boundary_w(itr) * sens_w(itr)
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
                logits = net(x, SENS[torch.as_tensor(b, device=dev)] if a.sens_in else None)
            loss = ord_loss(logits.float(), yb) if a.ordinal else -(target(yb) * F.log_softmax(logits.float(), 1)).sum(1).mean()
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
    pt = predict(net, te_idx)
    return va, pv, gap, best, pt


pos = np.cumsum(lab) - 1
oof = np.zeros((int(lab.sum()), NUM_CLASSES), np.float32)
fold_f1, gaps, covered = [], [], np.zeros(len(df), bool)
test_sum, n_folds_done = np.zeros((len(te_idx), NUM_CLASSES), np.float64), 0
for f in [int(x) for x in a.folds.split(",")]:
    va, pv, gap, mon, pt = run_fold(f)
    test_sum += pt
    n_folds_done += 1
    oof[pos[va]] = pv
    covered[va] = True
    fold_f1.append(mf1(y_all[va], pv.argmax(1)))
    gaps.append(gap)
    print(f"== fold {f}: outer F1 {fold_f1[-1]:.4f} (inner mon {mon:.3f}) gap {gap:.3f}", flush=True)

m = covered[lab]
yl, pl = y_all[lab][m], oof[m].argmax(1)
(ROOT / "outputs" / "preds").mkdir(parents=True, exist_ok=True)
np.save(ROOT / "outputs" / "preds" / f"ft_{name}_oof.npy", oof)
if not a.smoke:
    np.save(ROOT / "outputs" / "preds" / f"ft_{name}_test.npy", (test_sum / max(1, n_folds_done)).astype(np.float32))
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
