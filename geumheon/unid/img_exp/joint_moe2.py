"""이미지 + 센서 + 분류기 동시 학습 MoE v2.

v1(joint_moe.py) 대비 바뀐 점
  1. 센서 인코더: MLP → 인과적 트랜스포머 디코더. 같은 실험의 과거 --hist 행(현재 포함)의 센서 피처를
     토큰으로 쓰고, causal mask로 미래 토큰을 막는다. 위치 대신 '현재로부터 몇 초 전'(Δt)을 인코딩해
     행 간격이 다른 Test(10/15초)에도 같은 의미가 되게 한다. 출력은 현재(마지막) 토큰.
  2. 이미지: 224×224 강제 리사이즈 대신 원본 비율 유지. 짧은 변 ≥ --min_side(112px=8패치),
     긴 변 ≤ --max_side(336px=24패치), 각 변은 14의 배수, 최근접 보간(열화상 색=온도 값 보존).
     같은 크기끼리 배치를 묶어 패딩 없이 DINOv2에 넣는다(위치 임베딩은 HF가 크기별로 보간).
     원본 폭·높이·비율을 게이트 입력에 추가.
  3. 학습 20 에폭 (v1 10).
  4. GPU 메모리 한도 --mem_gb(기본 15GB)를 강제 → 학습이 끝까지 돌면 16GB GPU에서 돌아간다는 확인.

학습·검증 구성, 손실, 출력 형식은 v1과 같다 (JOINT_MOE.md 참고).
    python joint_moe2.py --data_dir <data> --out joint_moe2_s_u4 [--folds 0,3]
    python joint_moe2.py --data_dir <data> --out joint_moe2_s_u4 --merge_only
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from PIL import Image
from sklearn.model_selection import GroupKFold

import joint_moe as J1
from finetune import F, MEAN, STD, log
from finetune_dino import K

PATCH = 14
G = {}  # 전역 데이터 (DataLoader 워커가 fork로 공유): buf, off, shp, prev


# ---------------------------------------------------------------------------
# 이미지: 비율 유지 + 패치 수 범위 + 최근접 보간
# ---------------------------------------------------------------------------
def target_size(w, h, min_side, max_side):
    s = max(1.0, min_side / min(w, h))           # 작은 쪽이 min_side 이상이 되도록 확대 (축소는 안 함)
    if max(w, h) * s > max_side:                 # 긴 쪽이 max_side를 넘으면 줄임
        s = max_side / max(w, h)
    tw = max(PATCH, int(round(w * s / PATCH)) * PATCH)
    th = max(PATCH, int(round(h * s / PATCH)) * PATCH)
    return tw, th


def _size(p):
    with Image.open(p) as im:
        return im.size


def _load(a):
    p, tw, th = a
    im = Image.open(p).convert('RGB')
    w, h = im.size
    return np.asarray(im.resize((tw, th), Image.NEAREST), np.uint8), w, h


def load_all_var(data_dir, min_side, max_side):
    frames, paths = [], []
    for split in ['train', 'validation', 'test']:
        df = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
        df['split'] = split
        frames.append(df)
        paths += [str(data_dir / split / p) for p in df.image_path]
    df = pd.concat(frames, ignore_index=True)
    t0 = time.time()
    with Pool(8) as pool:
        wh = np.array(pool.map(_size, paths, chunksize=256))
    # 실험마다 목표 크기 하나 (가장 흔한 원본 크기 기준) → 현재/30초 전 프레임 크기가 항상 같음
    tgt = np.zeros((len(df), 2), int)
    for _, idx in df.groupby(['split', 'experiment_id'], sort=False).indices.items():
        u, c = np.unique(wh[idx], axis=0, return_counts=True)
        tgt[idx] = target_size(*u[c.argmax()], min_side, max_side)
    with Pool(8) as pool:
        res = pool.map(_load, [(p, tw, th) for p, (tw, th) in zip(paths, tgt)], chunksize=128)
    shp = np.array([r[0].shape for r in res], np.int64)                    # (h, w, 3)
    orig = wh.astype(np.float32)                                           # 원본 (w, h)
    sizes = shp.prod(1)
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    buf = np.empty(sizes.sum(), np.uint8)
    for i, r in enumerate(res):
        buf[off[i]:off[i] + sizes[i]] = r[0].ravel()
    del res
    log(f'cached {len(df)} frames, {buf.nbytes/1e9:.1f} GB, {len(np.unique(shp[:, :2], axis=0))} sizes in {time.time()-t0:.0f}s')
    t_all = df.relative_time_s.to_numpy(float)
    prev = np.zeros(len(df), int)
    for _, idx in df.groupby(['split', 'experiment_id'], sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        t = t_all[idx]
        j = np.clip(np.searchsorted(t, t - 30, side='right') - 1, 0, None)
        prev[idx] = idx[j]
    assert (shp[prev] == shp).all(), 'current/prev frame size mismatch'
    return df, buf, off, shp, orig, prev


def get_img(i):
    h, w, c = G['shp'][i]
    return G['buf'][G['off'][i]:G['off'][i] + h * w * c].reshape(h, w, c)


class VarDS(torch.utils.data.Dataset):
    def __init__(self, rows, labels, train):
        self.rows, self.labels, self.train = rows, labels, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        i = self.rows[k]
        x = np.concatenate([get_img(i), get_img(G['prev'][i])], 2)        # h×w×6
        if self.train:
            h, w = x.shape[:2]
            s = np.random.uniform(0.85, 1.0)
            ch, cw = max(1, int(h * s)), max(1, int(w * s))
            y0, x0 = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
            yy = y0 + (np.arange(h) * ch // h)                              # 최근접 보간으로 원래 크기 복원
            xx = x0 + (np.arange(w) * cw // w)
            x = x[yy][:, xx]
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).contiguous()
        y = -1 if self.labels is None else int(self.labels[i]) - 1
        return x, y, i


def bucket_batches(rows, bs, shuffle, rng=None):
    """같은 (h, w)끼리 배치. 학습 때는 버킷 안과 배치 순서를 섞는다."""
    key = G['shp'][rows, 0] * 10000 + G['shp'][rows, 1]
    out = []
    for kv in np.unique(key):
        r = np.where(key == kv)[0]
        if shuffle:
            r = rng.permutation(r)
        out += [r[j:j + bs].tolist() for j in range(0, len(r), bs)]
    if shuffle:
        out = [out[j] for j in rng.permutation(len(out))]
    return out


def to_input(x):
    x = x.float() / 255.0
    m, s = MEAN.to(x.device), STD.to(x.device)
    return torch.cat([(x[:, :3] - m) / s, (x[:, 3:] - m) / s], 1)


# ---------------------------------------------------------------------------
# 센서: 인과적 트랜스포머 디코더
# ---------------------------------------------------------------------------
def build_hist(df, L):
    """각 행마다 같은 실험의 과거 L행(현재 포함, 시간순, 왼쪽 -1 패딩) 전역 인덱스."""
    H = np.full((len(df), L), -1, np.int64)
    t = df.relative_time_s.to_numpy(float)
    for _, idx in df.groupby(['split', 'experiment_id'], sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        for r in range(len(idx)):
            seg = idx[max(0, r - L + 1):r + 1]
            H[idx[r], L - len(seg):] = seg
    return H


class SensorDecoder(nn.Module):
    def __init__(self, in_dim, L, d=128, layers=3, heads=4, out=256):
        super().__init__()
        self.L, self.heads = L, heads
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_dim, d), nn.GELU())
        self.tenc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.tx = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, out), nn.GELU(), nn.Dropout(0.1))
        self.register_buffer('causal', torch.triu(torch.ones(L, L, dtype=torch.bool), 1), persistent=False)
        self.register_buffer('eye', torch.eye(L, dtype=torch.bool), persistent=False)

    def forward(self, Xp, T, hist, cur):
        pad = hist < 0
        j = hist.clamp(min=0)
        dt = (T[cur][:, None] - T[j]).clamp(min=0)
        x = self.inp(Xp[j]) + self.tenc(torch.stack([dt / 300, torch.log1p(dt / 5), pad.float()], -1))
        # True = 차단: 미래 토큰 + 패딩 토큰. 자기 자신은 항상 허용(전부 차단된 행의 NaN 방지)
        m = (self.causal[None] | pad[:, None, :]) & ~self.eye[None]
        h = self.tx(x, mask=m.repeat_interleave(self.heads, 0))
        return self.out(self.norm(h[:, -1]))


class JointMoE2(nn.Module):
    def __init__(self, name, n_unfreeze, sensor_dim, L, hid=256):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(name)
        layers = self.backbone.encoder.layer
        self.n_layers = len(layers)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in layers[self.n_layers - n_unfreeze:]:
            for p in blk.parameters():
                p.requires_grad = True
        if n_unfreeze > 0:
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True
        d = self.backbone.config.hidden_size
        self.img_proj = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(0.1), nn.Linear(3 * d, hid), nn.GELU())
        self.sen_enc = SensorDecoder(sensor_dim, L, out=hid)
        self.head_s = nn.Linear(hid, K)
        self.head_i = nn.Linear(hid, K)
        self.head_j = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, K))
        self.gate = nn.Sequential(nn.Linear(2 * hid + 3, 64), nn.GELU(), nn.Linear(64, 3 * K))

    def forward(self, x6, ctx, cur):
        b = x6.shape[0]
        h = self.backbone(pixel_values=torch.cat([x6[:, :3], x6[:, 3:]], 0)).last_hidden_state
        cls, pm = h[:, 0], h[:, 1:].mean(1)
        zi = self.img_proj(torch.cat([cls[:b], pm[:b], cls[:b] - cls[b:]], 1))
        zs = self.sen_enc(ctx['Xp'], ctx['T'], ctx['H'][cur], cur)
        zz = torch.cat([zs, zi], 1)
        S, I, J = self.head_s(zs), self.head_i(zi), self.head_j(zz)
        w = self.gate(torch.cat([zz, ctx['SZ'][cur]], 1)).view(b, 3, K).softmax(1)
        return (w * torch.stack([S, I, J], 1)).sum(1), S, I, w


# ---------------------------------------------------------------------------
# 학습 / 예측
# ---------------------------------------------------------------------------
def make_loader(rows, labels, bs, train, rng=None):
    batches = bucket_batches(rows, bs, train, rng)
    ds = VarDS(rows, labels, train)
    return torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=6, pin_memory=True)


def predict(model, rows, ctx, dev, bs=128):
    n = len(rows)
    inv = {r: k for k, r in enumerate(rows)}
    out = {'cum': np.zeros((n, K), np.float32), 'cum_s': np.zeros((n, K), np.float32),
           'cum_i': np.zeros((n, K), np.float32), 'w': np.zeros((n, 3), np.float32)}
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for x, _, i in make_loader(np.asarray(rows), None, bs, False):
            cur = i.to(dev)
            o, S, I, w = model(to_input(x.to(dev, non_blocking=True)), ctx, cur)
            pos = np.array([inv[int(r)] for r in i])
            out['cum'][pos] = torch.sigmoid(o.float()).cpu().numpy()
            out['cum_s'][pos] = torch.sigmoid(S.float()).cpu().numpy()
            out['cum_i'][pos] = torch.sigmoid(I.float()).cpu().numpy()
            out['w'][pos] = w.float().mean(2).cpu().numpy()
    return out


def train_fold(args, df, tr_idx, ev_idx, ctx_base, dev, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    labels = df.label.to_numpy(float)
    prep = J1.SensorPrep(J1.XS[tr_idx])
    ctx = dict(ctx_base, Xp=torch.from_numpy(prep(J1.XS)).to(dev))   # 전 행 전처리 (fold 학습 통계 사용)
    y_tr = labels[tr_idx].astype(int)
    pos = np.array([(y_tr >= k).mean() for k in range(2, 7)])
    pw = torch.tensor(np.sqrt((1 - pos) / np.maximum(pos, 1e-3)), dtype=torch.float32, device=dev)
    model = JointMoE2(args.model, args.n_unfreeze, prep.dim, args.hist).to(dev)
    groups = J1.param_groups(model, args)
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    exps = df.experiment_id.to_numpy()
    g_rows = {e: tr_idx[exps[tr_idx] == e] for e in np.unique(exps[tr_idx])}
    n_ep = sum(min(len(v), args.per_exp) for v in g_rows.values())
    steps = int(n_ep / args.bs * 1.3) + 40                                  # 버킷 자투리 배치 여유분
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=args.epochs * steps,
                                                pct_start=0.1, anneal_strategy='cos')
    ev_cap = F.cap_rows(df.iloc[ev_idx]).index.to_numpy()
    ev_frame = df.iloc[ev_cap].reset_index(drop=True)
    ks = torch.arange(2, 7, device=dev).view(1, K)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f'  trainable {sum(p.numel() for p in trainable)/1e6:.1f}M, sensor dim {prep.dim}, hist {args.hist}')
    bce = lambda z, t: Fnn.binary_cross_entropy_with_logits(z, t, pos_weight=pw)
    done = 0
    for ep in range(args.epochs):
        rows = np.concatenate([rng.choice(v, min(len(v), args.per_exp), replace=False) for v in g_rows.values()])
        model.train(); tl, n, t0 = 0.0, 0, time.time()
        for x, y, i in make_loader(rows, labels, args.bs, True, rng):
            x, y, cur = to_input(x.to(dev, non_blocking=True)), y.to(dev), i.to(dev)
            tgt = ((y.view(-1, 1) + 1) >= ks).float() * (1 - args.ls) + 0.5 * args.ls
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o, S, I, _ = model(x, ctx, cur)
            loss = bce(o.float(), tgt) + args.aux_w * (bce(S.float(), tgt) + bce(I.float(), tgt))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 2.0)
            opt.step()
            if done < args.epochs * steps - 1:
                sched.step(); done += 1
            tl += loss.item() * len(y); n += len(y)
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            R = predict(model, ev_cap, ctx, dev)
            log(f'  ep{ep+1:02d} loss {tl/n:.4f}  fold F1(post) moe {J1.fold_f1(ev_frame, R["cum"]):.4f} '
                f'sensor {J1.fold_f1(ev_frame, R["cum_s"]):.4f} image {J1.fold_f1(ev_frame, R["cum_i"]):.4f}  '
                f'w(s,i,j) {R["w"].mean(0).round(2).tolist()}  ({time.time()-t0:.0f}s, '
                f'peak GPU reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB)')
        else:
            log(f'  ep{ep+1:02d} loss {tl/n:.4f}  ({time.time()-t0:.0f}s, peak {torch.cuda.max_memory_reserved()/2**30:.1f} GiB)')
    return model, prep, ctx, ev_cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--model', default='facebook/dinov2-small')
    ap.add_argument('--n_unfreeze', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--llrd', type=float, default=0.8)
    ap.add_argument('--head_lr', type=float, default=1e-3)
    ap.add_argument('--aux_w', type=float, default=0.3)
    ap.add_argument('--ls', type=float, default=0.05)
    ap.add_argument('--per_exp', type=int, default=200)
    ap.add_argument('--hist', type=int, default=64, help='센서 디코더가 보는 과거 행 수 (현재 포함)')
    ap.add_argument('--min_side', type=int, default=112)
    ap.add_argument('--max_side', type=int, default=336)
    ap.add_argument('--mem_gb', type=float, default=15.0, help='PyTorch GPU 메모리 한도 (16GB GPU 재현)')
    ap.add_argument('--eval_every', type=int, default=2)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--folds', default=None)
    ap.add_argument('--merge_only', action='store_true')
    args = ap.parse_args()
    log(vars(args))
    keys = ('cum', 'cum_s', 'cum_i', 'w')
    parts = args.out.parent / f'{args.out.stem}_parts'
    if args.merge_only:
        frames = [pd.read_csv(args.data_dir / sp / f'{sp}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'}).assign(split=sp)
                  for sp in ('train', 'validation', 'test')]
        df = pd.concat(frames, ignore_index=True)
        return J1.merge(args, df, np.where(df.split == 'train')[0], np.where(df.split != 'train')[0], keys, parts)

    dev = 'cuda'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / total))
    log(f'GPU memory cap {args.mem_gb} GiB of {total/2**30:.1f} GiB')
    ckpt_dir = args.out.parent / f'{args.out.stem}_models'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    parts.mkdir(parents=True, exist_ok=True)

    df, buf, off, shp, orig, prev = load_all_var(args.data_dir, args.min_side, args.max_side)
    G.update(buf=buf, off=off, shp=shp, prev=prev)
    feats = [F.build_features(df[df.split == s]) for s in ('train', 'validation', 'test')]
    J1.XS = pd.concat(feats).loc[df.sample_id.astype(str).values].to_numpy(np.float32)
    H = build_hist(df, args.hist)
    sz = np.stack([orig[:, 0] / 336, orig[:, 1] / 336, np.log(orig[:, 0] / orig[:, 1])], 1).astype(np.float32)
    ctx_base = dict(T=torch.tensor(df.relative_time_s.to_numpy(np.float32), device=dev),
                    H=torch.from_numpy(H).to(dev), SZ=torch.from_numpy(sz).to(dev))
    log(f'sensor {J1.XS.shape}, hist {H.shape}')

    tr = np.where(df.split == 'train')[0]
    others = np.where(df.split != 'train')[0]
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=df.experiment_id.to_numpy()[tr]))
    run = range(5) if args.folds is None else [int(f) for f in args.folds.split(',')]
    for k in run:
        a, b = splits[k]
        log(f'=== fold {k} ===')
        model, prep, ctx, ev_cap = train_fold(args, df, tr[a], tr[b], ctx_base, dev, seed=k)
        Ro = predict(model, tr[b], ctx, dev)
        Rx = predict(model, others, ctx, dev)
        np.savez(parts / f'fold_{k}.npz', oof_idx=tr[b], ev_cap=ev_cap,
                 **{f'oof_{kk}': Ro[kk] for kk in keys}, **{f'oth_{kk}': Rx[kk] for kk in keys})
        torch.save({'args': vars(args), 'fold': k, 'state_dict': model.state_dict(),
                    'prep': {'mu': prep.mu, 'sd': prep.sd, 'nan_cols': prep.nan_cols}}, ckpt_dir / f'fold_{k}.pt')
        log(f'fold {k} done, peak GPU reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB')
        del model, ctx; torch.cuda.empty_cache()
    if all((parts / f'fold_{k}.npz').exists() for k in range(5)):
        J1.merge(args, df, tr, others, keys, parts)


if __name__ == '__main__':
    main()
