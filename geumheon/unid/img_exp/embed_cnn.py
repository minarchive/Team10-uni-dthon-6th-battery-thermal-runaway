"""고정(frozen) ImageNet CNN으로 모든 열화상 프레임의 특징 벡터를 추출한다 (학습 없음, 추론만).

    python embed_cnn.py --data_dir <battery_data_light> --model resnet50.a1_in1k --out emb_resnet50.npz
    python embed_cnn.py --data_dir <battery_data_light> --model efficientnet_b0.ra_in1k --out emb_effb0.npz

- 224×224 리사이즈, ImageNet 평균/표준편차 정규화, 마지막 특징맵 global average pooling
  (ResNet50 2048차원, EfficientNet-B0 1280차원), float16 저장
- 프레임 한 장씩 독립 처리라 누수 없음. 4차원 압축(PCA)은 학습 단계(mm_transformer.py)에서 fold 학습 실험으로만 fit
- 가중치: timm `resnet50.a1_in1k`, `efficientnet_b0.ra_in1k` (ImageNet-1k, Apache-2.0), 캐시 hf_cache/
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from embed import Frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--model', default='resnet50.a1_in1k')
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    dev = 'cuda'
    model = timm.create_model(a.model, pretrained=True, num_classes=0).to(dev).eval().to(memory_format=torch.channels_last)
    ids, paths = [], []
    for split in ['train', 'validation', 'test']:
        df = pd.read_csv(a.data_dir / split / f'{split}.csv', dtype={'sample_id': 'string'}, usecols=['sample_id', 'image_path'])
        ids += df.sample_id.tolist()
        paths += [str(a.data_dir / split / p) for p in df.image_path]
    print(f'{a.model}: {len(paths)} frames, feature dim {model.num_features}', flush=True)
    dl = DataLoader(Frames(paths, a.size), batch_size=a.bs, num_workers=6, pin_memory=True)
    feats, t0 = [], time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, x in enumerate(dl):
            f = model(x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)).float()
            feats.append(f.cpu().numpy().astype(np.float16))
            if i % 40 == 0:
                print(f'{i * a.bs}/{len(paths)} {time.time() - t0:.0f}s', flush=True)
    np.savez(a.out, sample_id=np.array(ids), feat=np.concatenate(feats))
    print(f'saved {a.out} in {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)


if __name__ == '__main__':
    main()
