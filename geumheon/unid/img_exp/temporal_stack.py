"""시간 스택 EfficientNet-B0: 여러 시점 열화상을 채널로 쌓아 CNN 이 시간 변화를 직접 보게 한다.

입력 24채널 (RGB × 8, 같은 실험의 현재·과거 프레임만):
  [현재 RGB, 현재−첫 프레임, 현재−10초 전, 현재−20초 전, 현재−30초 전, 현재−60초 전, 현재−120초 전, 현재−300초 전] 각 RGB 차이
  (과거 프레임이 없으면 첫 프레임). SOTA 3채널 흑백 차이 입력(첫/15/60초)의 확장.
  흑백을 쓰지 않는 이유: 열화상이 의사 컬러(채도 ~255)라 RGB 평균 밝기가 온도와 단조 관계가 아님 (28°C 126 → 359°C 96).
백본: torchvision EfficientNet-B0 ImageNet-1K (stem conv 3→24채널: 현재 RGB 는 사전학습 가중치 그대로, 차이 7묶음은 그 가중치/7)
학습: SOTA train_vision 과 같은 설정 (AdamW 3e-4, OneCycle, 8 epoch, bs 64, 클래스 가중치, label smoothing 0.05,
      좌우 뒤집기, 224~256 랜덤 리사이즈) + 5/10/15초 간격 격자 무작위 재현 (Test 다운샘플 대비)

평가 (final_oc4 와 같은 fold·프레임):
  CV : Train GroupKFold(5, shuffle, seed 0), 평가는 500행 캡
  Val: Train 전체 학습 → Val 500행 캡
  점수: 이미지 단독, oc4 R_J6wide 센서 확률(final_oc4_gpu/oof·valprobs)과 결합 (이미지 비중 0.06/0.1/0.2), 후처리 τ 0.7
Test: Train+Val 학습 → oc4 센서 Test 확률과 0.1 결합 → 제출 파일

  python temporal_stack.py cv | val | test | score
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as TF
import torchvision.models as tvm
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from final_oc import F, load, D, score  # noqa: E402

OUT = HERE / 'temporal_stack'
(OUT / 'ckpt').mkdir(parents=True, exist_ok=True)
LAGS = [10, 20, 30, 60, 120, 300]
NG = 2 + len(LAGS)                    # 묶음 수: 현재 + 첫 프레임 차이 + lag 차이
K = 3 * NG                            # RGB
IMG = 256
SEED = 42
dev = 'cuda'
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def seed_all(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ 프레임 인덱스 (df 행 위치 기준)
def _groups(df):
    t = df.relative_time_s.to_numpy(float)
    for _, idx in df.reset_index(drop=True).groupby('experiment_id', sort=False).indices.items():
        yield idx[np.argsort(t[idx], kind='stable')], t


def first_index(df):
    out = np.arange(len(df))
    for idx, _ in _groups(df):
        out[idx] = idx[0]
    return out


def past_index(df, lag, stride=1):
    """t−lag 이하 가장 최근 프레임. stride>1 이면 실험을 stride 간격 격자로 본 것처럼 (Test 다운샘플 재현)."""
    out = np.arange(len(df))
    for idx, t in _groups(df):
        for o in range(stride):
            q = idx[o::stride]; tq = t[q]
            out[q] = q[np.maximum(np.searchsorted(tq, tq - lag, side='right') - 1, 0)]
    return out


# ------------------------------------------------------------------ 이미지
def load_rgb(df, root):
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    def ld(p):
        with Image.open(Path(root) / p) as im:
            return np.asarray(im.convert('RGB').resize((IMG, IMG), Image.BICUBIC), dtype=np.uint8)
    with ThreadPoolExecutor(16) as ex:
        return np.stack(list(ex.map(ld, df.image_path.astype(str).tolist())))


def make_input(I, rows, f0, lag_rows, flip=None):
    """I: (N,H,W,3) uint8 RGB. rows/f0: (B,), lag_rows: (L,B) → (B,K,H,W) 정규화 텐서"""
    g = lambda r: torch.from_numpy(I[r]).to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.
    cur = g(rows)
    ch = [(cur - MEAN.to(dev)) / STD.to(dev), (cur - g(f0)) * 4]
    ch += [(cur - g(lr)) * 4 for lr in lag_rows]
    x = torch.cat(ch, 1)
    if flip is not None:
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    return x


class StackNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = tvm.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        c = m.features[0][0]
        new = nn.Conv2d(K, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(torch.cat([c.weight] + [c.weight / (NG - 1)] * (NG - 1), 1))
        m.features[0][0] = new
        d = m.classifier[1].in_features; m.classifier = nn.Identity()
        self.net, self.head = m, nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 6))

    def forward(self, x):
        return self.head(self.net(x))


def train(df, I, seed, epochs=8, bs=64, lr=3e-4):
    seed_all(seed); rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    y = df.label.to_numpy(int) - 1; f0 = first_index(df)
    G = np.stack([np.stack([past_index(df, lag, s) for lag in LAGS]) for s in (1, 2, 3)])   # (3, L, N)
    net = StackNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * (len(df) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    cw = torch.tensor(len(y) / (6 * np.bincount(y, minlength=6)), dtype=torch.float32, device=dev)
    for ep in range(epochs):
        net.train(); perm = rng.permutation(len(df)); tl = n = 0; t0 = time.time()
        for b in range(0, len(perm) - bs + 1, bs):
            r = perm[b:b + bs]; k = rng.integers(3, size=len(r))
            lag_rows = G[k, :, r].T                                  # (L, B)
            flip = torch.rand(len(r), device=dev) < 0.5
            x = make_input(I, r, f0[r], lag_rows, flip)
            s = int(rng.integers(224, 257)); x = TF.interpolate(x, size=(s, s), mode='bilinear', align_corners=False)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = net(x).float()
            loss = TF.cross_entropy(z, torch.from_numpy(y[r]).to(dev), weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item(); n += 1
        log(f'  ep{ep} loss {tl / n:.4f} ({time.time() - t0:.0f}s)')
    return net


@torch.no_grad()
def predict(net, I, fr):
    """fr 행 순서 = I 행 순서. 이전 프레임 인덱스는 fr(평가 프레임) 안에서 계산."""
    fr = fr.reset_index(drop=True); f0 = first_index(fr)
    LR = np.stack([past_index(fr, lag) for lag in LAGS])
    net.eval(); out = []
    for b in range(0, len(fr), 256):
        r = np.arange(b, min(b + 256, len(fr)))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(torch.softmax(net(make_input(I, r, f0[r], LR[:, r])).float(), 1).cpu().numpy())
    return np.concatenate(out)


def get_net(name, df, I, seed=SEED):
    ck = OUT / 'ckpt' / f'{name}.pt'
    if ck.exists():
        net = StackNet(pretrained=False).to(dev); net.load_state_dict(torch.load(ck, map_location=dev))
        log(f'loaded {ck.name}'); return net
    log(f'train {name} ({len(df)} rows)')
    net = train(df, I, seed)
    torch.save(net.state_dict(), ck)
    return net


def frames():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    for f in (tr, va, te):
        f['sample_id'] = f.sample_id.astype(str)
    return tr, va, te, F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)


def rows_of(frame, sub):
    return pd.Series(np.arange(len(frame)), index=frame.sample_id.values).loc[sub.sample_id.values].to_numpy()


def run_cv():
    tr, va, te, tr_cap, va_cap = frames()
    p = OUT / 'oof_tr_cap.npy'
    if p.exists():
        log('cv exists'); return
    I = load_rgb(tr, D / 'train'); log(f'train images {I.shape}')
    P = np.full((len(tr_cap), 6), np.nan, np.float32)
    for k, (a, b) in enumerate(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id)):
        net = get_net(f'cv_fold{k}', tr.iloc[a], I[a])
        m = ~tr_cap.experiment_id.isin(set(tr.experiment_id.iloc[a])).to_numpy(); ev = tr_cap[m]
        P[m] = predict(net, I[rows_of(tr, ev)], ev)
        log(f'fold{k} done'); del net; torch.cuda.empty_cache()
    assert np.isfinite(P).all()
    np.save(p, P)


def run_val():
    tr, va, te, tr_cap, va_cap = frames()
    p = OUT / 'val_va_cap.npy'
    if p.exists():
        log('val exists'); return
    I = load_rgb(tr, D / 'train')
    net = get_net('val_train_all', tr, I); del I
    Iv = load_rgb(va, D / 'validation')
    np.save(p, predict(net, Iv[rows_of(va, va_cap)], va_cap))
    log('val done')


def run_test():
    tr, va, te, tr_cap, va_cap = frames()
    p = OUT / 'test.npy'
    if not p.exists():
        tv = pd.concat([tr, va], ignore_index=True)
        I = np.concatenate([load_rgb(tr, D / 'train'), load_rgb(va, D / 'validation')])
        net = get_net('final_trval', tv, I); del I
        np.save(p, predict(net, load_rgb(te, D / 'test'), te)); log('test probs done')
    Pv = np.load(p)
    pS = np.load(HERE / 'final_oc4_gpu' / 'test_probs_R_J6wide.npy')
    ss = pd.read_csv(D / 'sample_submission.csv', dtype={'sample_id': str})
    for w in (0.1, 0.2):
        lab = F.postprocess(te, (1 - w) * pS + w * Pv, 0.7)
        sub = ss[['sample_id']].merge(pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab}), on='sample_id', how='left')
        assert sub.label.notna().all() and len(sub) == len(ss)
        sub.label = sub.label.astype(int)
        q = OUT / f'submission_R_J6wide_tstack{w}.csv'; sub.to_csv(q, index=False)
        log(f'wrote {q.name} {sub.label.value_counts().sort_index().to_dict()}')


def run_score():
    tr, va, te, tr_cap, va_cap = frames()
    S_cv = np.load(HERE / 'final_oc4_gpu' / 'oof_R_J6wide.npy'); S_va = np.load(HERE / 'final_oc4_gpu' / 'valprobs_R_J6wide.npy')
    from final_oc2 import cnn
    C_cv = cnn('ft_convnext_tiny.npz', tr_cap.sample_id.values); C_va = cnn('ft_convnext_tiny.npz', va_cap.sample_id.values)
    V_cv = np.load(OUT / 'oof_tr_cap.npy'); V_va = np.load(OUT / 'val_va_cap.npy')
    res = {}

    def rep(nm, pc, pv):
        cv, cv_oc, cpc = score(tr_cap, pc); v, v_oc, vpc = score(va_cap, pv)
        res[nm] = dict(cv=cv, cv_oc=cv_oc, val=v, val_oc=v_oc, cv_per_class=cpc, val_per_class=vpc)
        log(f'{nm:24s} CV {cv:.4f} OC {cv_oc:.4f} | Val {v:.4f} OC {v_oc:.4f} | Val {vpc}')
    rep('img_only_tstack', V_cv, V_va)
    rep('img_only_convnext', C_cv, C_va)
    rep('sensor_only', S_cv, S_va)
    for w in (0.06, 0.1, 0.2):
        rep(f'oc4+tstack{w}', (1 - w) * S_cv + w * V_cv, (1 - w) * S_va + w * V_va)
    rep('oc4+convnext0.1 (기준)', 0.9 * S_cv + 0.1 * C_cv, 0.9 * S_va + 0.1 * C_va)
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    {'cv': run_cv, 'val': run_val, 'test': run_test, 'score': run_score}[sys.argv[1]]()
