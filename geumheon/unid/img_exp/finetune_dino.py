"""DINOv2 부분 파인튜닝 + 순서형(cumulative) 헤드 → 이미지 전문가 OOF.

구조  : 현재 프레임과 30초 전 프레임을 같은 DINOv2 백본에 통과 (가중치 공유).
        특징 = [CLS_cur, patch_mean_cur, CLS_cur − CLS_prev] → LayerNorm → Linear(5)
        5개 출력 = logit P(y ≥ k), k = 2..6 (순서형). 클래스 확률은 누적확률 차분으로 복원.
부분 FT: 패치 임베딩과 앞쪽 블록은 고정, 마지막 --n_unfreeze 개 블록 + 최종 LayerNorm + 헤드만 학습.
        블록별 layer-wise lr decay(뒤쪽일수록 큰 lr).
검증  : finetune.py와 동일 — Train 78실험 GroupKFold(5, shuffle, seed 0), fold 검증은 500행 캡 후 평가.
출력  : <out>.npz — sample_id, probs(n×6), cum(n×5: P(≥2..6)), split.
        Train은 OOF, Validation·Test는 5개 fold 모델 평균.

    python finetune_dino.py --data_dir <battery_data_light> --model facebook/dinov2-small --n_unfreeze 4 --out ft_ord_dinov2s_u4
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from finetune import DS, F, MEAN, SIZE, STD, load_all, log

K = 5  # 경계 수 (≥2 .. ≥6)
SENS = None  # residual 모드: 전체 df 행 순서의 센서 logit P(≥k) (n×5), main에서 채움


class DSI(DS):
    def __getitem__(self, k):
        x, y = super().__getitem__(k)
        return x, y, self.rows[k]


def sens_of(idx, dev):
    return None if SENS is None else torch.tensor(SENS[idx.numpy()], dtype=torch.float32, device=dev)


class OrdDino(nn.Module):
    def __init__(self, name, n_unfreeze, residual=False):
        super().__init__()
        self.residual = residual
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
        self.norm = nn.LayerNorm(3 * d)
        self.head = nn.Linear(3 * d, K)
        if residual:
            # 게이트: 이미지 특징 + 센서 누적확률(확신도 문맥) → 경계별 이미지 보정 가중치
            self.gate = nn.Sequential(nn.Linear(3 * d + K, 64), nn.GELU(), nn.Linear(64, K))
            nn.init.zeros_(self.gate[-1].weight); nn.init.constant_(self.gate[-1].bias, -2.0)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.drop = nn.Dropout(0.1)

    def encode(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state   # B×(1+N)×D (마지막 LayerNorm 적용됨)
        return h[:, 0], h[:, 1:].mean(1)

    def forward(self, x6, S=None):
        b = x6.shape[0]
        cls, pm = self.encode(torch.cat([x6[:, :3], x6[:, 3:]], 0))
        c_cur, c_prev, pm_cur = cls[:b], cls[b:], pm[:b]
        z = self.norm(torch.cat([c_cur, pm_cur, c_cur - c_prev], 1))
        z = self.drop(z)
        if not self.residual:
            return self.head(z)                                # B×5 cumulative logits
        I = self.head(z)
        g = torch.sigmoid(self.gate(torch.cat([z, torch.sigmoid(S)], 1)))
        return S + g * I                                       # 센서 logit + 게이트·이미지 보정


def to_input(x):
    """uint8 B×6×H×W → 두 프레임 모두 ImageNet 정규화 (DINOv2 전처리와 동일)."""
    x = x.float() / 255.0
    if x.shape[-1] != SIZE:
        x = Fnn.interpolate(x, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
    m, s = MEAN.to(x.device), STD.to(x.device)
    return torch.cat([(x[:, :3] - m) / s, (x[:, 3:] - m) / s], 1)


def cum_to_probs(cum):
    """P(≥k) (n×5) → 단조 보정 후 클래스 확률 (n×6)."""
    c = np.minimum.accumulate(np.clip(cum, 0, 1), axis=1)
    full = np.concatenate([np.ones((len(c), 1)), c, np.zeros((len(c), 1))], 1)
    return np.clip(full[:, :-1] - full[:, 1:], 0, 1).astype(np.float32)


def predict(model, ds, dev, bs=256):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=4, pin_memory=True)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for x, _, i in dl:
            out.append(torch.sigmoid(model(to_input(x.to(dev, non_blocking=True)), sens_of(i, dev)).float()).cpu().numpy())
    return np.concatenate(out)


def param_groups(model, args):
    head = list(model.norm.parameters()) + list(model.head.parameters())
    if model.residual:
        head += list(model.gate.parameters())
    groups = [{'params': head, 'lr': args.head_lr}]
    L = model.n_layers
    for i in range(L - args.n_unfreeze, L):
        lr = args.lr * args.llrd ** (L - 1 - i)
        groups.append({'params': list(model.backbone.encoder.layer[i].parameters()), 'lr': lr})
    if args.n_unfreeze > 0:
        groups.append({'params': list(model.backbone.layernorm.parameters()), 'lr': args.lr})
    return groups


def train_fold(args, df, imgs, prev, tr_idx, ev_idx, dev, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    labels = df.label.to_numpy(float)
    y_tr = labels[tr_idx].astype(int)
    # 경계별 양성 가중치 (sqrt 완화) — 희소한 상위 단계 경계(≥5, ≥6)를 덜 무시하도록
    pos = np.array([(y_tr >= k).mean() for k in range(2, 7)])
    pw = torch.tensor(np.sqrt((1 - pos) / np.maximum(pos, 1e-3)), dtype=torch.float32, device=dev)
    if SENS is not None:  # 잔차 모드: 센서 확률의 보정을 유지하려고 가중치 없음
        pw = torch.ones_like(pw)
    model = OrdDino(args.model, args.n_unfreeze, residual=SENS is not None).to(dev)
    groups = param_groups(model, args)
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    exps = df.experiment_id.to_numpy()
    g_rows = {e: tr_idx[exps[tr_idx] == e] for e in np.unique(exps[tr_idx])}
    steps = sum(min(len(v), args.per_exp) for v in g_rows.values()) // args.bs + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=args.epochs * steps,
                                                pct_start=0.15, anneal_strategy='cos')
    ev_cap = F.cap_rows(df.iloc[ev_idx]).index.to_numpy()
    ev_ds = DSI(ev_cap, imgs, prev, labels, train=False)
    ks = torch.arange(2, 7, device=dev).view(1, K)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f'  trainable params {sum(p.numel() for p in trainable)/1e6:.1f}M / {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
    for ep in range(args.epochs):
        rows = np.concatenate([np.random.choice(v, min(len(v), args.per_exp), replace=False) for v in g_rows.values()])
        dl = torch.utils.data.DataLoader(DSI(rows, imgs, prev, labels, train=True), batch_size=args.bs, shuffle=True,
                                         num_workers=6, pin_memory=True, drop_last=True)
        model.train(); tl, n, t0 = 0.0, 0, time.time()
        for x, y, i in dl:
            x, y, S = to_input(x.to(dev, non_blocking=True)), y.to(dev), sens_of(i, dev)
            tgt = ((y.view(-1, 1) + 1) >= ks).float()                      # y는 0-based
            tgt = tgt * (1 - args.ls) + 0.5 * args.ls
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x, S).float()
            loss = Fnn.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 2.0)
            opt.step(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        C = predict(model, ev_ds, dev)
        P = cum_to_probs(C)
        ev_frame = df.iloc[ev_cap].reset_index(drop=True)
        yv = ev_frame.label.astype(int).to_numpy()
        f_raw = f1_score(yv, F.LABELS[P.argmax(1)], labels=F.LABELS, average='macro')
        f_post = f1_score(yv, F.postprocess(ev_frame, P, 0.7), labels=F.LABELS, average='macro')
        log(f'  ep{ep+1:02d} loss {tl/n:.4f}  fold F1 raw {f_raw:.4f} post {f_post:.4f}  ({time.time()-t0:.0f}s)')
    return model, ev_cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--model', default='facebook/dinov2-small')
    ap.add_argument('--n_unfreeze', type=int, default=4, help='학습할 마지막 transformer 블록 수 (0=선형 프로브)')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-5, help='마지막 블록 lr (앞 블록은 llrd 배씩 감소)')
    ap.add_argument('--llrd', type=float, default=0.8)
    ap.add_argument('--head_lr', type=float, default=1e-3)
    ap.add_argument('--ls', type=float, default=0.05)
    ap.add_argument('--per_exp', type=int, default=150)
    ap.add_argument('--sensor_residual', action='store_true',
                    help='센서(LightGBM) 확률을 고정 offset으로 두고 융합 출력에 손실 (Train=OOF, Val/Test=전체 모델)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    log(vars(args))
    dev = 'cuda'
    ckpt_dir = args.out.parent / f'{args.out.stem}_models'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    df, imgs, prev = load_all(args.data_dir)
    if args.sensor_residual:
        global SENS
        zo, zv = np.load('seq_base_oof.npz'), np.load('seq_base_valtest.npz')
        P = pd.concat([pd.DataFrame(zo['probs'], index=zo['sample_id'].astype(str)),
                       pd.DataFrame(zv['val_probs'], index=zv['val_sample_id'].astype(str)),
                       pd.DataFrame(zv['test_probs'], index=zv['test_sample_id'].astype(str))]).loc[df.sample_id.astype(str)].to_numpy(float)
        C = np.clip(np.cumsum(P[:, ::-1], 1)[:, ::-1][:, 1:], 1e-4, 1 - 1e-4)
        SENS = np.log(C / (1 - C)).astype(np.float32)
        log(f'sensor residual mode: SENS {SENS.shape}')
    tr = np.where(df.split == 'train')[0]
    others = np.where(df.split != 'train')[0]
    cum = np.full((len(df), K), np.nan, np.float32)
    acc_other = np.zeros((len(others), K), np.float32)
    oof_frames = []
    for k, (a, b) in enumerate(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=df.experiment_id.to_numpy()[tr])):
        log(f'=== fold {k} ===')
        model, ev_cap = train_fold(args, df, imgs, prev, tr[a], tr[b], dev, seed=k)
        cum[tr[b]] = predict(model, DSI(tr[b], imgs, prev, None, False), dev)
        oof_frames.append(ev_cap)
        acc_other += predict(model, DSI(others, imgs, prev, None, False), dev) / 5
        sd = {n: p for n, p in model.state_dict().items()}
        torch.save({'model_name': args.model, 'n_unfreeze': args.n_unfreeze, 'fold': k, 'state_dict': sd},
                   ckpt_dir / f'fold_{k}.pt')
        del model; torch.cuda.empty_cache()
    cum[others] = acc_other
    probs = cum_to_probs(cum)
    ev = np.concatenate(oof_frames)
    frame = df.iloc[ev].reset_index(drop=True)
    y = frame.label.astype(int).to_numpy()
    P = probs[ev]
    post = F.postprocess(frame, P, 0.7)
    log(f'CV (img only, cap, OOF) raw {f1_score(y, F.LABELS[P.argmax(1)], labels=F.LABELS, average="macro"):.4f} '
        f'post {f1_score(y, post, labels=F.LABELS, average="macro"):.4f} '
        f'per-class {f1_score(y, post, labels=F.LABELS, average=None).round(3).tolist()}')
    va = np.where(df.split == 'validation')[0]
    vf = df.iloc[va].reset_index(drop=True)
    vc = F.cap_rows(vf)
    Pv = probs[va][vc.index.to_numpy()]
    vc = vc.reset_index(drop=True)
    yv = vc.label.astype(int).to_numpy()
    log(f'VAL (img only, cap, 5-fold avg) raw {f1_score(yv, F.LABELS[Pv.argmax(1)], labels=F.LABELS, average="macro"):.4f} '
        f'post {f1_score(yv, F.postprocess(vc, Pv, 0.7), labels=F.LABELS, average="macro"):.4f}')
    np.savez(args.out.with_suffix('.npz'), sample_id=df.sample_id.to_numpy().astype(str), probs=probs, cum=cum,
             split=df.split.to_numpy().astype(str))
    log(f'saved {args.out.with_suffix(".npz")}')


if __name__ == '__main__':
    main()
