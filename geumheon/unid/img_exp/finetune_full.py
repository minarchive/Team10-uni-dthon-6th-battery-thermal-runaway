"""ConvNeXt-Tiny(timm ImageNet 사전학습) 를 Train+Val 전체로 파인튜닝 → Test 프레임 확률.

설정은 finetune.py(CV용)와 동일: 6채널 입력(현재 + 30초 전 대비 변화), 6에폭, AdamW lr 1e-4, OneCycle,
클래스 가중치(sqrt 완화) CE + label smoothing 0.1, 실험별 에폭당 최대 150프레임.
에폭 선택 없음(마지막 에폭 사용). train_fold 의 에폭별 로그는 학습 데이터(Val) 위 점수라 참고용 아님.
Test는 추론에만 사용. 시드별 체크포인트 저장.

    python finetune_full.py --data_dir <battery_data_light> --seeds 0 1 2
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import finetune as FT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--model', default='convnext_tiny')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--ls', type=float, default=0.1)
    ap.add_argument('--drop_path', type=float, default=0.1)
    ap.add_argument('--per_exp', type=int, default=150)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--out', type=Path, default=Path('ft_convnext_tiny_trval'))
    args = ap.parse_args()
    FT.log(vars(args))
    ckdir = args.out.parent / f'{args.out.stem}_models'
    ckdir.mkdir(parents=True, exist_ok=True)
    df, imgs, prev = FT.load_all(args.data_dir)
    lab = np.where(df.split != 'test')[0]
    va = np.where(df.split == 'validation')[0]
    te = np.where(df.split == 'test')[0]
    P = np.zeros((len(te), 6), np.float32)
    for s in args.seeds:
        ck = ckdir / f'seed_{s}.pt'
        model = FT.build(args.model, args.drop_path).to('cuda').to(memory_format=torch.channels_last)
        if ck.exists():
            model.load_state_dict(torch.load(ck, map_location='cuda')['state_dict'])
            FT.log(f'loaded {ck}')
        else:
            FT.log(f'=== seed {s} (train+val {len(lab)} rows) ===')
            model, _, _ = FT.train_fold(args, df, imgs, prev, lab, va, 'cuda', seed=s)
            torch.save({'model_name': args.model, 'num_classes': 6, 'in_chans': 6, 'drop_path': args.drop_path,
                        'seed': s, 'input_size': FT.SIZE, 'state_dict': model.state_dict()}, ck)
            FT.log(f'saved {ck}')
        P += FT.predict(model, FT.DS(te, imgs, prev, None, False), 'cuda') / len(args.seeds)
        del model; torch.cuda.empty_cache()
    np.savez(args.out.with_suffix('.npz'), sample_id=df.sample_id.to_numpy()[te].astype(str), probs=P)
    FT.log(f'saved {args.out.with_suffix(".npz")}  test pred dist {np.bincount(P.argmax(1) + 1, minlength=7)[1:].tolist()}')
    FT.log('ALL DONE')


if __name__ == '__main__':
    main()
