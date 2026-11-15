"""학습: Train + Validation 전체로 최종 모델 학습 후 weights/ 에 저장.
  1) LightGBM A  : 시계열 피처 전체 (일반 실험용)
  2) LightGBM C  : 압력 절대값 피처 제외 (과충전 tr_method==2 실험용)
  3) EfficientNet-B0 (실험 기준 이미지 입력) — seed별
  4) config.json : 이미지 혼합 비중 α (5-fold OOF 우도 최대화로 추정한 값, README 참조)
python train.py --data_dir /path/to/battery_data_light
"""
import argparse, json, time
from pathlib import Path
import joblib, lightgbm as lgb, numpy as np, pandas as pd, torch, torch.nn.functional as F
import features as FT
from model import (LGB_PARAMS, VisionNet, cols_without_abs_pressure, first_index, load_images, past_index_grid, seed_everything, vision_input)


def train_vision(df, images, seed, dev, epochs=8, bs=64, lr=3e-4):
    seed_everything(seed); rng = np.random.default_rng(seed)
    y = df.label.values.astype(int) - 1; f0 = first_index(df)
    G15 = np.stack([past_index_grid(df, 15, k) for k in (1, 2, 3)]); G60 = np.stack([past_index_grid(df, 60, k) for k in (1, 2, 3)])
    net = VisionNet().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * (len(df) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    cw = torch.tensor(len(y) / (6 * np.bincount(y, minlength=6)), dtype=torch.float32, device=dev); t0 = time.time()
    for ep in range(epochs):
        net.train(); perm = rng.permutation(len(df)); tl = n = 0
        for b in range(0, len(perm) - bs + 1, bs):
            r = perm[b:b + bs]; k = rng.integers(3, size=len(r)); flip = torch.rand(len(r), device=dev) < 0.5
            x = vision_input(images, r, f0[r], G15[k, r], G60[k, r], dev, flip)
            s = int(rng.integers(224, 257)); x = F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                z = net(x).float()
            loss = F.cross_entropy(z, torch.from_numpy(y[r]).to(dev), weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item(); n += 1
        print(f'  vision seed{seed} ep{ep} loss {tl / n:.4f} {time.time() - t0:.0f}s', flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True); ap.add_argument('--out_dir', default='weights')
    ap.add_argument('--vision_seeds', default='42'); ap.add_argument('--vision_epochs', type=int, default=8); ap.add_argument('--alpha_trm1', type=float, default=0.1); ap.add_argument('--alpha_trm2', type=float, default=0.1)
    a = ap.parse_args(); D = Path(a.data_dir); out = Path(a.out_dir); out.mkdir(exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; seed_everything()
    tr = pd.read_csv(D / 'train' / 'train.csv'); va = pd.read_csv(D / 'validation' / 'validation.csv')
    df = pd.concat([tr, va], ignore_index=True); y = df.label.values.astype(int)
    X = FT.build(df, 'rich'); cols_c = cols_without_abs_pressure(X.columns)
    mA = lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y); mC = lgb.LGBMClassifier(**LGB_PARAMS).fit(X[cols_c], y)
    joblib.dump(dict(A=mA, C=mC, cols=list(X.columns), cols_c=cols_c), out / 'lgbm.joblib'); print(f'LightGBM 저장: 피처 {X.shape[1]} / 과충전용 {len(cols_c)}')
    images = np.concatenate([load_images(tr, D / 'train'), load_images(va, D / 'validation')])
    seeds = [int(s) for s in a.vision_seeds.split(',')]
    for s in seeds:
        net = train_vision(df, images, s, dev, epochs=a.vision_epochs); torch.save(net.state_dict(), out / f'vision_seed{s}.pt')
    json.dump(dict(vision_seeds=seeds, alpha_trm1=a.alpha_trm1, alpha_trm2=a.alpha_trm2), open(out / 'config.json', 'w'), indent=2)
    print('저장 완료:', sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    main()
