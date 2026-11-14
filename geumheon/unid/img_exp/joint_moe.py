"""이미지 + 센서 + 분류기를 한 번에 학습하는 MoE (end-to-end).

LightGBM은 미분이 안 되므로 센서 쪽을 MLP로 바꾸고, 전부 하나의 손실로 함께 학습한다.

  이미지 인코더: DINOv2 (마지막 --n_unfreeze 블록만 학습). 현재 + 30초 전 프레임 → [CLS, 패치평균, CLS 변화] → 256
  센서 인코더 : 센서 피처 735개 (val0.9195 features.build_features, 인과적) + 결측 표시 → MLP → 256
  전문가 3개  : 센서 헤드 / 이미지 헤드 / 결합 헤드, 각각 순서형 5 logit = logit P(y≥k), k=2..6
  게이트      : [z_sensor, z_image] → 경계별 softmax(3) → 세 전문가 logit의 가중합
  손실        : BCE(게이트 출력) + aux_w × (BCE(센서 헤드) + BCE(이미지 헤드))  — 각 전문가가 혼자서도 쓸 만하게 유지

학습·검증 구성은 finetune_dino.py와 동일 (Train 78실험 GroupKFold(5, seed 0), fold 검증 500행 캡, 확신 cummax τ=0.7).
OOF·Val·Test 출력이 모두 같은 모델에서 나오므로 스태킹에서 생기는 학습/추론 분포 차이가 없다.

    python joint_moe.py --data_dir <battery_data_light> --out joint_moe_s_u4
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

from finetune import DS, F, load_all, log
from finetune_dino import K, cum_to_probs, to_input

XS = None  # 전처리 전 센서 피처 (n×D, float32, NaN 포함). main에서 채움


class DSJ(DS):
    def __getitem__(self, k):
        x, y = super().__getitem__(k)
        return x, y, self.rows[k]


class SensorPrep:
    """fold 학습 행으로만 fit: 부호 보존 log 압축 → 표준화 → ±5 클리핑, NaN→0 + 결측 표시 열."""

    def __init__(self, X):
        Z = np.sign(X) * np.log1p(np.abs(X))
        self.mu = np.nanmean(Z, 0)
        self.sd = np.nanstd(Z, 0) + 1e-6
        self.mu = np.nan_to_num(self.mu); self.sd = np.nan_to_num(self.sd, nan=1.0)
        self.nan_cols = np.where(np.isnan(X).any(0))[0]

    def __call__(self, X):
        Z = np.sign(X) * np.log1p(np.abs(X))
        Z = np.clip((Z - self.mu) / self.sd, -5, 5)
        miss = np.isnan(X[:, self.nan_cols]).astype(np.float32)
        return np.concatenate([np.nan_to_num(Z, nan=0.0), miss], 1).astype(np.float32)

    @property
    def dim(self):
        return len(self.mu) + len(self.nan_cols)


class JointMoE(nn.Module):
    def __init__(self, name, n_unfreeze, sensor_dim, hid=256):
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
        self.sen_enc = nn.Sequential(nn.Linear(sensor_dim, 512), nn.GELU(), nn.Dropout(0.2),
                                     nn.Linear(512, hid), nn.GELU(), nn.Dropout(0.1))
        self.head_s = nn.Linear(hid, K)
        self.head_i = nn.Linear(hid, K)
        self.head_j = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, K))
        self.gate = nn.Sequential(nn.Linear(2 * hid, 64), nn.GELU(), nn.Linear(64, 3 * K))

    def forward(self, x6, xs):
        b = x6.shape[0]
        h = self.backbone(pixel_values=torch.cat([x6[:, :3], x6[:, 3:]], 0)).last_hidden_state
        cls, pm = h[:, 0], h[:, 1:].mean(1)
        zi = self.img_proj(torch.cat([cls[:b], pm[:b], cls[:b] - cls[b:]], 1))
        zs = self.sen_enc(xs)
        zz = torch.cat([zs, zi], 1)
        S, I, J = self.head_s(zs), self.head_i(zi), self.head_j(zz)
        w = self.gate(zz).view(b, 3, K).softmax(1)                     # 경계별 전문가 가중치
        out = (w * torch.stack([S, I, J], 1)).sum(1)
        return out, S, I, w


def param_groups(model, args):
    fast = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    groups = [{'params': fast, 'lr': args.head_lr}]
    L = model.n_layers
    for i in range(L - args.n_unfreeze, L):
        groups.append({'params': list(model.backbone.encoder.layer[i].parameters()), 'lr': args.lr * args.llrd ** (L - 1 - i)})
    if args.n_unfreeze > 0:
        groups.append({'params': list(model.backbone.layernorm.parameters()), 'lr': args.lr})
    return groups


def predict(model, ds, prep, dev, bs=256):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=4, pin_memory=True)
    out = {k: [] for k in ('cum', 'cum_s', 'cum_i', 'w')}
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for x, _, i in dl:
            xs = torch.from_numpy(prep(XS[i.numpy()])).to(dev)
            o, S, I, w = model(to_input(x.to(dev, non_blocking=True)), xs)
            out['cum'].append(torch.sigmoid(o.float()).cpu().numpy())
            out['cum_s'].append(torch.sigmoid(S.float()).cpu().numpy())
            out['cum_i'].append(torch.sigmoid(I.float()).cpu().numpy())
            out['w'].append(w.float().mean(2).cpu().numpy())         # 전문가별 평균 가중치 (n×3)
    return {k: np.concatenate(v) for k, v in out.items()}


def fold_f1(frame, C):
    y = frame.label.astype(int).to_numpy()
    return f1_score(y, F.postprocess(frame, cum_to_probs(C), 0.7), labels=F.LABELS, average='macro')


def train_fold(args, df, imgs, prev, tr_idx, ev_idx, dev, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    labels = df.label.to_numpy(float)
    prep = SensorPrep(XS[tr_idx])
    y_tr = labels[tr_idx].astype(int)
    pos = np.array([(y_tr >= k).mean() for k in range(2, 7)])
    pw = torch.tensor(np.sqrt((1 - pos) / np.maximum(pos, 1e-3)), dtype=torch.float32, device=dev)
    model = JointMoE(args.model, args.n_unfreeze, prep.dim).to(dev)
    groups = param_groups(model, args)
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    exps = df.experiment_id.to_numpy()
    g_rows = {e: tr_idx[exps[tr_idx] == e] for e in np.unique(exps[tr_idx])}
    steps = sum(min(len(v), args.per_exp) for v in g_rows.values()) // args.bs + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=args.epochs * steps,
                                                pct_start=0.15, anneal_strategy='cos')
    ev_cap = F.cap_rows(df.iloc[ev_idx]).index.to_numpy()
    ev_ds = DSJ(ev_cap, imgs, prev, labels, train=False)
    ev_frame = df.iloc[ev_cap].reset_index(drop=True)
    ks = torch.arange(2, 7, device=dev).view(1, K)
    trainable = [p for p in model.parameters() if p.requires_grad]
    bce = lambda z, t: Fnn.binary_cross_entropy_with_logits(z, t, pos_weight=pw)
    for ep in range(args.epochs):
        rows = np.concatenate([np.random.choice(v, min(len(v), args.per_exp), replace=False) for v in g_rows.values()])
        dl = torch.utils.data.DataLoader(DSJ(rows, imgs, prev, labels, train=True), batch_size=args.bs, shuffle=True,
                                         num_workers=6, pin_memory=True, drop_last=True)
        model.train(); tl, n, t0 = 0.0, 0, time.time()
        for x, y, i in dl:
            x, y = to_input(x.to(dev, non_blocking=True)), y.to(dev)
            xs = torch.from_numpy(prep(XS[i.numpy()])).to(dev)
            tgt = ((y.view(-1, 1) + 1) >= ks).float() * (1 - args.ls) + 0.5 * args.ls
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o, S, I, _ = model(x, xs)
            loss = bce(o.float(), tgt) + args.aux_w * (bce(S.float(), tgt) + bce(I.float(), tgt))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 2.0)
            opt.step(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        R = predict(model, ev_ds, prep, dev)
        log(f'  ep{ep+1:02d} loss {tl/n:.4f}  fold F1(post) moe {fold_f1(ev_frame, R["cum"]):.4f} '
            f'sensor {fold_f1(ev_frame, R["cum_s"]):.4f} image {fold_f1(ev_frame, R["cum_i"]):.4f}  '
            f'w(s,i,j) {R["w"].mean(0).round(2).tolist()}  ({time.time()-t0:.0f}s, '
            f'peak GPU reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB)')
    return model, prep, ev_cap


def report(tag, frame, R):
    y = frame.label.astype(int).to_numpy()
    parts = []
    for k in ('cum', 'cum_s', 'cum_i'):
        post = F.postprocess(frame, cum_to_probs(R[k]), 0.7)
        parts.append(f'{k} {f1_score(y, post, labels=F.LABELS, average="macro"):.4f}')
    post = F.postprocess(frame, cum_to_probs(R['cum']), 0.7)
    log(f'{tag}: ' + '  '.join(parts) + f'  | moe per-class {f1_score(y, post, labels=F.LABELS, average=None).round(3).tolist()}'
        f'  w(s,i,j) {R["w"].mean(0).round(3).tolist()}')


def main():
    global XS
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--model', default='facebook/dinov2-small')
    ap.add_argument('--n_unfreeze', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--llrd', type=float, default=0.8)
    ap.add_argument('--head_lr', type=float, default=1e-3)
    ap.add_argument('--aux_w', type=float, default=0.3)
    ap.add_argument('--ls', type=float, default=0.05)
    ap.add_argument('--per_exp', type=int, default=200)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--folds', default=None, help='이 프로세스가 학습할 fold (예: "0,3"). 기본 전부')
    ap.add_argument('--merge_only', action='store_true', help='학습 없이 _parts/fold_*.npz 합치기만')
    args = ap.parse_args()
    log(vars(args))
    dev = 'cuda'
    ckpt_dir = args.out.parent / f'{args.out.stem}_models'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        frames = [pd.read_csv(args.data_dir / sp / f'{sp}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'}).assign(split=sp)
                  for sp in ('train', 'validation', 'test')]
        df = pd.concat(frames, ignore_index=True)
        tr, others = np.where(df.split == 'train')[0], np.where(df.split != 'train')[0]
        parts = args.out.parent / f'{args.out.stem}_parts'
        return merge(args, df, tr, others, ('cum', 'cum_s', 'cum_i', 'w'), parts)
    df, imgs, prev = load_all(args.data_dir)
    t0 = time.time()
    feats = [F.build_features(df[df.split == s]) for s in ('train', 'validation', 'test')]
    XS = pd.concat(feats).loc[df.sample_id.astype(str).values].to_numpy(np.float32)
    log(f'sensor features {XS.shape} in {time.time()-t0:.0f}s')

    tr = np.where(df.split == 'train')[0]
    others = np.where(df.split != 'train')[0]
    keys = ('cum', 'cum_s', 'cum_i', 'w')
    parts = args.out.parent / f'{args.out.stem}_parts'
    parts.mkdir(parents=True, exist_ok=True)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=df.experiment_id.to_numpy()[tr]))
    run = range(5) if args.folds is None else [int(f) for f in args.folds.split(',')]
    if not args.merge_only:
        for k in run:
            a, b = splits[k]
            log(f'=== fold {k} ===')
            model, prep, ev_cap = train_fold(args, df, imgs, prev, tr[a], tr[b], dev, seed=k)
            Ro = predict(model, DSJ(tr[b], imgs, prev, None, False), prep, dev)
            Rx = predict(model, DSJ(others, imgs, prev, None, False), prep, dev)
            np.savez(parts / f'fold_{k}.npz', oof_idx=tr[b], ev_cap=ev_cap,
                     **{f'oof_{kk}': Ro[kk] for kk in keys}, **{f'oth_{kk}': Rx[kk] for kk in keys})
            torch.save({'args': vars(args), 'fold': k, 'state_dict': model.state_dict(),
                        'prep': {'mu': prep.mu, 'sd': prep.sd, 'nan_cols': prep.nan_cols}}, ckpt_dir / f'fold_{k}.pt')
            del model; torch.cuda.empty_cache()
    if not all((parts / f'fold_{k}.npz').exists() for k in range(5)):
        log('not all folds finished; skip merge')
        return
    merge(args, df, tr, others, keys, parts)


def merge(args, df, tr, others, keys, parts):
    R_all = {k: np.full((len(df), 3 if k == 'w' else K), np.nan, np.float32) for k in keys}
    acc = {k: np.zeros((len(others), 3 if k == 'w' else K), np.float32) for k in keys}
    oof_frames = []
    # 같은 fold의 LightGBM 센서 OOF와 비교 (seq_base_oof.npz, 같은 GroupKFold 분할)
    zo = np.load('seq_base_oof.npz')
    lgb = pd.DataFrame(zo['probs'], index=zo['sample_id'].astype(str))
    for k in range(5):
        z = np.load(parts / f'fold_{k}.npz')
        for kk in keys:
            R_all[kk][z['oof_idx']] = z[f'oof_{kk}']
            acc[kk] += z[f'oth_{kk}'] / 5
        ev = z['ev_cap']
        oof_frames.append(ev)
        fr = df.iloc[ev].reset_index(drop=True)
        y = fr.label.astype(int).to_numpy()
        Pl = lgb.loc[fr.sample_id.astype(str)].to_numpy()
        f_l = f1_score(y, F.postprocess(fr, Pl, 0.7), labels=F.LABELS, average='macro')
        log(f'fold {k}: moe {fold_f1(fr, R_all["cum"][ev]):.4f} sensor-MLP {fold_f1(fr, R_all["cum_s"][ev]):.4f} '
            f'image {fold_f1(fr, R_all["cum_i"][ev]):.4f} | LightGBM {f_l:.4f}')
    for kk in keys:
        R_all[kk][others] = acc[kk]
    ev = np.concatenate(oof_frames)
    report('CV  (Train OOF, cap)', df.iloc[ev].reset_index(drop=True), {k: v[ev] for k, v in R_all.items()})
    va = np.where(df.split == 'validation')[0]
    vc = F.cap_rows(df.iloc[va])
    vi = vc.index.to_numpy()
    report('VAL (cap, 5-fold avg)', vc.reset_index(drop=True), {k: v[vi] for k, v in R_all.items()})
    np.savez(args.out.with_suffix('.npz'), sample_id=df.sample_id.to_numpy().astype(str), split=df.split.to_numpy().astype(str),
             probs=cum_to_probs(R_all['cum']), **R_all)
    log(f'saved {args.out.with_suffix(".npz")}')


if __name__ == '__main__':
    main()
