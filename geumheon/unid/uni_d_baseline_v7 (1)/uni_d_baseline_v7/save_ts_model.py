"""v3 시계열 모델(LightGBM, class_weight=balanced)을 Train으로 학습해 저장하고 Val로 재현 확인.

예) python save_ts_model.py --out model_ts_v3.joblib
    python save_ts_model.py --out model_ts_v3_trval.joblib --train_val   # 제출용: Train+Val로 학습
"""
import argparse
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

import exp_post as E
import report as R
import ts_model as M

TAU = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='/home/ubuntu/data/battery_data_light')
    ap.add_argument('--out', default='model_ts_v3.joblib')
    ap.add_argument('--train_val', action='store_true', help='Train+Validation 전체로 학습(검증 점수 없음)')
    args = ap.parse_args()

    tr, va = E.load(args.data_dir, 'train'), E.load(args.data_dir, 'validation')
    fit_frame = pd.concat([tr, va], ignore_index=True) if args.train_val else tr
    X = M.build_features(fit_frame)
    model = M.build_model('lgbm', 'balanced').fit(X, fit_frame.label)

    bundle = dict(model=model, features=list(X.columns), tau=TAU, class_weight='balanced',
                  trained_on='train+validation' if args.train_val else 'train',
                  versions=dict(lightgbm=lightgbm.__version__, sklearn=sklearn.__version__))
    joblib.dump(bundle, args.out)
    print('저장:', Path(args.out).resolve())

    if not args.train_val:   # 재현 확인 (Val 500행 캡)
        vc = M.cap_rows(va)
        P = model.predict_proba(M.build_features(vc)[bundle['features']])
        raw = E.L[P.argmax(1)]
        post = E._per_exp(vc, P, lambda P, t: E.conf_cummax(P, TAU))
        print(f'Val Macro F1  raw {E.f1(vc.label, raw):.4f} | 확신 cummax τ={TAU} {E.f1(vc.label, post):.4f}')
        R.print_tables(vc, post, f'Val — 저장 모델 + 확신 cummax τ={TAU}')


if __name__ == '__main__':
    main()
