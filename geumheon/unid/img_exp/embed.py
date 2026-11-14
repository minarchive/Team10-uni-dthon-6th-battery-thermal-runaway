"""고정(frozen) DINOv2로 모든 프레임의 임베딩을 추출합니다.

    python embed.py --data_dir <battery_data_light> --model facebook/dinov2-small --out emb_dinov2s.npz

저장: sample_id(문자열 배열), cls(n×d), patch_mean(n×d), float16
- 사전학습 가중치: Hugging Face `facebook/dinov2-small` / `dinov2-base` (Apache-2.0)
- 학습 없음, 추론만. 프레임 한 장씩 독립적으로 처리하므로 누수 없음.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Frames(Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert('RGB').resize((self.size, self.size), Image.BICUBIC)
        x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--model', default='facebook/dinov2-small')
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModel
    dev = 'cuda'
    model = AutoModel.from_pretrained(args.model).to(dev).eval()

    ids, paths = [], []
    for split in ['train', 'validation', 'test']:
        f = args.data_dir / split / f'{split}.csv'
        if not f.exists():
            print('skip', split); continue
        df = pd.read_csv(f, dtype={'sample_id': 'string'}, usecols=['sample_id', 'image_path'])
        ids += df.sample_id.tolist()
        paths += [str(args.data_dir / split / p) for p in df.image_path]
    print(f'{len(paths)} frames', flush=True)

    dl = DataLoader(Frames(paths, args.size), batch_size=args.bs, num_workers=6, pin_memory=True)
    cls, pm = [], []
    t0 = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, x in enumerate(dl):
            h = model(pixel_values=x.to(dev, non_blocking=True)).last_hidden_state.float()
            cls.append(h[:, 0].cpu().numpy().astype(np.float16))
            pm.append(h[:, 1:].mean(1).cpu().numpy().astype(np.float16))
            if i % 20 == 0:
                print(f'{i * args.bs}/{len(paths)} {time.time() - t0:.0f}s', flush=True)
    np.savez(args.out, sample_id=np.array(ids), cls=np.concatenate(cls), patch_mean=np.concatenate(pm))
    print(f'saved {args.out} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
