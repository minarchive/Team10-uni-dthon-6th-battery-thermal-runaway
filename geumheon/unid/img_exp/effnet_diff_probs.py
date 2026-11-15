"""v3_sota_et0.8920 의 EfficientNet-B0 (차이 영상 입력: 현재−첫/15초 전/60초 전) 확률 만들기.

  CV : Train GroupKFold(5, shuffle, seed 0) — final_oc4 와 같은 fold. fold 마다 SOTA train_vision(seed 42) 로 새로 학습
  Val: Train 전체로 학습 (seed 42)
  Test: v3_sota_et0.8920/weights/vision_seed{42,7}.pt (Train+Val 학습) 그대로, 두 seed 평균
평가 프레임은 실험당 500행 캡(F.cap_rows, Test 조건), 이전 프레임 인덱스도 캡 프레임 안에서 계산 (SOTA test.py 와 같음).

출력 (final_oc2.cnn 형식: sample_id, probs)
  eff_diff_b0.npz        : tr_cap(OOF) + va_cap
  eff_diff_b0_trval.npz  : test
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

SNAP = Path('/home/nxcolelxu/Team10-uni-dthon-6th-battery-thermal-runaway/v3_sota_et0.8920')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_oc import F, load, D  # noqa: E402  (exp_img 가 val0.9195/features 를 'features' 로 import)

# 스냅샷 model/train/features 는 이름이 겹치므로 잠시 모듈 캐시를 비우고 불러온 뒤 원래대로 되돌림
_saved = {k: sys.modules.pop(k, None) for k in ('features', 'model', 'train')}
sys.path.insert(0, str(SNAP))
import model as SM  # noqa: E402
from train import train_vision  # noqa: E402
sys.path.remove(str(SNAP))
for _k, _m in _saved.items():
    sys.modules.pop(_k, None)
    if _m is not None:
        sys.modules[_k] = _m

HERE = Path(__file__).resolve().parent
CK = HERE / 'eff_diff_b0_ckpt'; CK.mkdir(exist_ok=True)
T0 = time.time()
dev = 'cuda'


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def predict(nets, images, fr):
    """fr 행 순서 = images 행 순서. 이전 프레임 인덱스는 fr 안에서 계산."""
    fr = fr.reset_index(drop=True)
    f0, p15, p60 = SM.first_index(fr), SM.past_index(fr, 15), SM.past_index(fr, 60)
    P = 0
    for net in nets:
        net.eval(); out = []
        with torch.no_grad():
            for b in range(0, len(fr), 256):
                r = np.arange(b, min(b + 256, len(fr)))
                out.append(torch.softmax(net(SM.vision_input(images, r, f0[r], p15[r], p60[r], dev)).float(), 1).cpu().numpy())
        P = P + np.concatenate(out) / len(nets)
    return P


def get_net(name, df, images):
    ck = CK / f'{name}.pt'
    net = SM.VisionNet(pretrained=False).to(dev)
    if ck.exists():
        net.load_state_dict(torch.load(ck, map_location=dev)); log(f'loaded {ck.name}')
    else:
        net = train_vision(df.reset_index(drop=True), images, 42, dev)
        torch.save(net.state_dict(), ck); log(f'trained {ck.name}')
    return net


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    for f in (tr, va, te):
        f['sample_id'] = f.sample_id.astype(str)
    tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)

    # Test: 스냅샷 가중치 그대로
    out_te = HERE / 'eff_diff_b0_trval.npz'
    if not out_te.exists():
        img_te = SM.load_images(te, D / 'test')
        nets = []
        for s in (42, 7):
            n = SM.VisionNet(pretrained=False).to(dev)
            n.load_state_dict(torch.load(SNAP / 'weights' / f'vision_seed{s}.pt', map_location=dev)); nets.append(n)
        np.savez(out_te, sample_id=te.sample_id.values, probs=predict(nets, img_te, te).astype(np.float32))
        log('test done'); del nets, img_te

    out = HERE / 'eff_diff_b0.npz'
    if out.exists():
        log('exists', out.name); return
    img_tr = SM.load_images(tr, D / 'train'); img_va = SM.load_images(va, D / 'validation')
    pos_tr = pd.Series(np.arange(len(tr)), index=tr.sample_id.values)
    pos_va = pd.Series(np.arange(len(va)), index=va.sample_id.values)
    log(f'images {img_tr.shape} {img_va.shape}')
    P_tr = np.full((len(tr_cap), 6), np.nan, np.float32)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    for k, (a, b) in enumerate(splits):
        net = get_net(f'cv_fold{k}', tr.iloc[a], img_tr[a])
        m = ~tr_cap.experiment_id.isin(set(tr.experiment_id.iloc[a])).to_numpy()
        ev = tr_cap[m]
        P_tr[m] = predict([net], img_tr[pos_tr.loc[ev.sample_id.values].to_numpy()], ev)
        log(f'fold{k} done ({m.sum()} rows)'); del net; torch.cuda.empty_cache()
    assert np.isfinite(P_tr).all()
    net = get_net('val_train_all', tr, img_tr)
    P_va = predict([net], img_va[pos_va.loc[va_cap.sample_id.values].to_numpy()], va_cap)
    np.savez(out, sample_id=np.r_[tr_cap.sample_id.values, va_cap.sample_id.values], probs=np.r_[P_tr, P_va].astype(np.float32))
    log('saved', out.name)


if __name__ == '__main__':
    main()
