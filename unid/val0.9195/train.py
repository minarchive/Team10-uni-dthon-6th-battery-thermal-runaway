"""model.joblib 재학습 (같은 데이터·시드면 같은 모델이 나옵니다).

    python train.py --data_dir <데이터 경로> --out model.joblib              # Train만 (= 제공된 model.joblib)
    python train.py --data_dir <데이터 경로> --out model_trval.joblib --train_val   # Train+Validation (최종 제출용 후보)
"""
import argparse
from pathlib import Path

import joblib
import lightgbm
import pandas as pd
import sklearn

import features as F


def load(d, split):
    return pd.read_csv(Path(d) / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True, help='train/, validation/ 폴더가 있는 경로')
    ap.add_argument('--out', type=Path, default=Path('model.joblib'))
    ap.add_argument('--train_val', action='store_true')
    args = ap.parse_args()

    frame = load(args.data_dir, 'train')
    if args.train_val:
        frame = pd.concat([frame, load(args.data_dir, 'validation')], ignore_index=True)
    X = F.build_features(frame)
    model = F.build_model().fit(X, frame['label'].astype(int))
    joblib.dump(dict(model=model, features=list(X.columns), tau=0.7, class_weight='balanced',
                     trained_on='train+validation' if args.train_val else 'train',
                     versions=dict(lightgbm=lightgbm.__version__, sklearn=sklearn.__version__)), args.out)
    print(f'저장: {args.out} (학습 {len(frame)}행, 피처 {X.shape[1]}개)')


if __name__ == '__main__':
    main()
