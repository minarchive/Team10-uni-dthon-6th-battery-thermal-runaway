"""3단계: 소형 백본 파인튜닝 (timm, ImageNet 사전학습) → OOF 확률 생성.

입력  : 현재 프레임(RGB 3ch) + (현재 − 30초 전 프레임)(3ch) = 6채널. 30초 전 프레임이 없으면 첫 프레임 사용(과거만 사용).
검증  : Train 78개 실험 GroupKFold(5, seed 0). 각 fold 검증 실험은 500행 캡 적용 후 평가.
출력  : <out>.npz — sample_id, probs(n×6), split. Train은 OOF, Validation·Test는 5개 fold 모델 평균.
        <out stem>_models/fold_*.pt — 재사용 가능한 5개 fold CNN 체크포인트.
손실  : 클래스 가중치 CE + label smoothing. 증강: 약한 crop·이동·뒤집기만.
샘플링: 에폭마다 실험별 최대 N프레임을 무작위로 솎아 씀(거의 같은 프레임 반복 학습 방지).

    python finetune.py --data_dir <battery_data_light> --model convnext_tiny --out ft_convnext_tiny
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402

SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(*a):
    print(*a, flush=True)


def _load(p):
    return np.asarray(Image.open(p).convert('RGB').resize((SIZE, SIZE), Image.BICUBIC), np.uint8)


def load_all(data_dir):
    frames, paths = [], []
    for split in ['train', 'validation', 'test']:
        df = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
        df['split'] = split
        frames.append(df)
        paths += [str(data_dir / split / p) for p in df.image_path]
    df = pd.concat(frames, ignore_index=True)
    t0 = time.time()
    with Pool(6) as pool:
        imgs = np.stack(pool.map(_load, paths, chunksize=256))
    log(f'cached {imgs.shape} uint8 ({imgs.nbytes/1e9:.1f} GB) in {time.time()-t0:.0f}s')
    # 30초 전 프레임 인덱스 (같은 실험, 과거만)
    prev = np.zeros(len(df), int)
    t_all = df.relative_time_s.to_numpy(float)
    for _, idx in df.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        t = t_all[idx]
        j = np.clip(np.searchsorted(t, t - 30, side='right') - 1, 0, None)
        prev[idx] = idx[j]
    return df, imgs, prev


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, imgs, prev, labels, train):
        self.rows, self.imgs, self.prev, self.labels, self.train = rows, imgs, prev, labels, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        i = self.rows[k]
        cur, pre = self.imgs[i], self.imgs[self.prev[i]]
        x = np.concatenate([cur, pre], 2)                      # H×W×6 uint8
        if self.train:
            # 약한 증강: 같은 crop/이동/뒤집기를 두 프레임에 동일하게 적용
            s = np.random.uniform(0.85, 1.0)
            h = w = int(SIZE * s)
            y0, x0 = np.random.randint(0, SIZE - h + 1), np.random.randint(0, SIZE - w + 1)
            x = x[y0:y0 + h, x0:x0 + w]
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
        # 캐시 배열의 view 를 그대로 텐서로 만들면 worker→main 공유메모리 전달(collate)에서
        # "Trying to resize storage that is not resizable" 가 나므로 소유 메모리로 복사
        x = torch.tensor(np.ascontiguousarray(x)).permute(2, 0, 1).contiguous()   # 6×h×w
        if x.shape[-1] != SIZE or x.shape[-2] != SIZE:   # crop 뒤 크기 통일 (배치 stack 가능하도록)
            x = Fnn.interpolate(x[None].float(), size=(SIZE, SIZE), mode='bilinear', align_corners=False)[0].round().to(torch.uint8)
        y = -1 if self.labels is None else int(self.labels[i]) - 1
        return x, y


def to_input(x):
    x = x.float() / 255.0
    if x.shape[-1] != SIZE:
        x = Fnn.interpolate(x, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
    cur, pre = x[:, :3], x[:, 3:]
    cur = (cur - MEAN.to(x.device)) / STD.to(x.device)
    diff = (x[:, :3] - pre) / 0.25                              # 변화량 채널 (대략 단위 분산)
    return torch.cat([cur, diff], 1)


def build(model_name, drop_path):
    return timm.create_model(model_name, pretrained=True, num_classes=6, in_chans=6, drop_path_rate=drop_path)


def predict(model, ds, dev, bs=256):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=4, pin_memory=True)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for x, _ in dl:
            out.append(model(to_input(x.to(dev, non_blocking=True))).float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def train_fold(args, df, imgs, prev, tr_idx, ev_idx, dev, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    labels = df.label.to_numpy(float)
    y_tr = labels[tr_idx].astype(int)
    cnt = np.bincount(y_tr - 1, minlength=6).astype(float)
    cw = torch.tensor(np.sqrt(cnt.sum() / (6 * np.maximum(cnt, 1))), dtype=torch.float32, device=dev)  # sqrt 완화
    model = build(args.model, args.drop_path).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    exps = df.experiment_id.to_numpy()
    groups = {e: tr_idx[exps[tr_idx] == e] for e in np.unique(exps[tr_idx])}
    steps_per_epoch = sum(min(len(v), args.per_exp) for v in groups.values()) // args.bs + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.15, anneal_strategy='cos')
    ev_cap = F.cap_rows(df.iloc[ev_idx]).index.to_numpy()
    ev_ds = DS(ev_cap, imgs, prev, labels, train=False)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.ls)
    best, best_P = -1, None
    for ep in range(args.epochs):
        rows = np.concatenate([np.random.choice(v, min(len(v), args.per_exp), replace=False) for v in groups.values()])
        dl = torch.utils.data.DataLoader(DS(rows, imgs, prev, labels, train=True), batch_size=args.bs, shuffle=True,
                                         num_workers=6, pin_memory=True, drop_last=True)
        model.train(); tl, n, t0 = 0.0, 0, time.time()
        for x, y in dl:
            x, y = to_input(x.to(dev, non_blocking=True)).contiguous(memory_format=torch.channels_last), y.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        P = predict(model, ev_ds, dev)
        ev_frame = df.iloc[ev_cap].reset_index(drop=True)
        yv = ev_frame.label.astype(int).to_numpy()
        f_raw = f1_score(yv, F.LABELS[P.argmax(1)], labels=F.LABELS, average='macro')
        f_post = f1_score(yv, F.postprocess(ev_frame, P, 0.7), labels=F.LABELS, average='macro')
        log(f'  ep{ep+1:02d} loss {tl/n:.4f}  fold F1 raw {f_raw:.4f} post {f_post:.4f}  ({time.time()-t0:.0f}s)')
    return model, ev_cap, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--model', default='convnext_tiny')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--ls', type=float, default=0.1)
    ap.add_argument('--drop_path', type=float, default=0.1)
    ap.add_argument('--per_exp', type=int, default=150, help='에폭당 실험별 최대 프레임 수')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--checkpoint_dir', type=Path, default=None, help='fold별 CNN 가중치 저장 디렉터리')
    args = ap.parse_args()
    log(vars(args))
    dev = 'cuda'
    checkpoint_dir = args.checkpoint_dir or args.out.parent / f'{args.out.stem}_models'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    df, imgs, prev = load_all(args.data_dir)
    tr = np.where(df.split == 'train')[0]
    others = np.where(df.split != 'train')[0]
    probs = np.full((len(df), 6), np.nan, np.float32)
    acc_other = np.zeros((len(others), 6), np.float32)
    oof_frames = []
    for k, (a, b) in enumerate(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=df.experiment_id.to_numpy()[tr])):
        log(f'=== fold {k} ===')
        model, ev_cap, P = train_fold(args, df, imgs, prev, tr[a], tr[b], dev, seed=k)
        # OOF: 캡 안 된 행 포함 전부 예측(스태킹 피처용), 평가는 캡된 행으로
        P_all = predict(model, DS(tr[b], imgs, prev, None, False), dev)
        probs[tr[b]] = P_all
        oof_frames.append(ev_cap)
        acc_other += predict(model, DS(others, imgs, prev, None, False), dev) / 5
        checkpoint = checkpoint_dir / f'fold_{k}.pt'
        tmp_checkpoint = checkpoint.with_name(checkpoint.name + '.tmp')
        torch.save({
            'model_name': args.model,
            'num_classes': 6,
            'in_chans': 6,
            'drop_path': args.drop_path,
            'fold': k,
            'seed': k,
            'input_size': SIZE,
            'state_dict': model.state_dict(),
        }, tmp_checkpoint)
        tmp_checkpoint.replace(checkpoint)
        log(f'saved fold {k} checkpoint: {checkpoint}')
        del model; torch.cuda.empty_cache()
    probs[others] = acc_other
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
    np.savez(args.out.with_suffix('.npz'), sample_id=df.sample_id.to_numpy().astype(str), probs=probs,
             split=df.split.to_numpy().astype(str))
    log(f'saved {args.out.with_suffix(".npz")}')
    log(f'saved CNN fold checkpoints in {checkpoint_dir}')


if __name__ == '__main__':
    main()
