"""Extra Trees 학습 (Train + Validation 전체). LightGBM과 같은 피처·과충전 분기(A: 전체 피처, C: 압력 절대값 제외).
최종 모델은 leaf 크기가 다른 ET 두 개를 쓴다:
  python train_et.py --data_dir $DATA --leaf 20 --out weights/et.joblib
  python train_et.py --data_dir $DATA --leaf 40 --out weights/et40.joblib"""
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import features as FT
from model import cols_without_abs_pressure, SEED

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', required=True); ap.add_argument('--leaf', type=int, default=20); ap.add_argument('--out', default=None); ap.add_argument('--n_jobs', type=int, default=16); a = ap.parse_args()
    out = Path(a.out or f'weights/et{"" if a.leaf == 20 else a.leaf}.joblib'); out.parent.mkdir(parents=True, exist_ok=True)
    ET_PARAMS = dict(n_estimators=500, min_samples_leaf=a.leaf, max_features=0.3, n_jobs=a.n_jobs, random_state=SEED)
    D = Path(a.data_dir); df = pd.concat([pd.read_csv(D / 'train' / 'train.csv'), pd.read_csv(D / 'validation' / 'validation.csv')], ignore_index=True)
    y = df.label.values.astype(int); X = FT.build(df, 'rich'); cols = list(X.columns); cols_c = cols_without_abs_pressure(cols)
    A = ExtraTreesClassifier(**ET_PARAMS).fit(X[cols].values.astype(np.float32), y); C = ExtraTreesClassifier(**ET_PARAMS).fit(X[cols_c].values.astype(np.float32), y)
    joblib.dump(dict(A=A, C=C, cols=cols, cols_c=cols_c), out, compress=3); print(f'Extra Trees(leaf={a.leaf}) 저장: {out}')

if __name__ == '__main__':
    main()
