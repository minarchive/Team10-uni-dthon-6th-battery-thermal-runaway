"""Extra Trees 학습 (Train + Validation 전체) → weights/et.joblib. LightGBM과 같은 피처·과충전 분기(A: 전체 피처, C: 압력 절대값 제외).
python train_et.py --data_dir /path/to/battery_data_light"""
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import features as FT
from model import cols_without_abs_pressure, SEED

ET_PARAMS = dict(n_estimators=500, min_samples_leaf=20, max_features=0.3, n_jobs=48, random_state=SEED)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', required=True); ap.add_argument('--out_dir', default='weights'); a = ap.parse_args()
    D = Path(a.data_dir); df = pd.concat([pd.read_csv(D / 'train' / 'train.csv'), pd.read_csv(D / 'validation' / 'validation.csv')], ignore_index=True)
    y = df.label.values.astype(int); X = FT.build(df, 'rich'); cols = list(X.columns); cols_c = cols_without_abs_pressure(cols)
    A = ExtraTreesClassifier(**ET_PARAMS).fit(X[cols].values.astype(np.float32), y); C = ExtraTreesClassifier(**ET_PARAMS).fit(X[cols_c].values.astype(np.float32), y)
    joblib.dump(dict(A=A, C=C, cols=cols, cols_c=cols_c), Path(a.out_dir) / 'et.joblib', compress=3); print('Extra Trees 저장')

if __name__ == '__main__':
    main()
