"""시계열 전용 모델 학습·평가.

예) python train_ts.py --data_dir ./data/battery_data_light --model_out ./model_ts.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

import ts_model as M


def load_split(data_dir: Path, split: str):
    frame = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    X = M.build_features(frame)
    y = pd.to_numeric(frame['label'], errors='raise').astype(int).to_numpy()
    return frame, X, y


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=M.LABELS, average='macro', zero_division=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--model_out', type=Path, default=Path('model_ts.joblib'))
    p.add_argument('--refit_train_val', action='store_true')
    args = p.parse_args()

    train, X_tr, y_tr = load_split(args.data_dir, 'train')
    val, X_va, y_va = load_split(args.data_dir, 'validation')
    assert set(train.experiment_id).isdisjoint(val.experiment_id)
    groups = train.experiment_id.to_numpy()
    print(f'피처 수: {X_tr.shape[1]} | Train {len(X_tr)}행 / Validation {len(X_va)}행')

    configs = [('lgbm', None), ('lgbm', 'balanced')]
    results = {}
    n_splits = min(5, len(np.unique(groups)))
    for kind, cw in configs:
        name = f'{kind}/{cw or "none"}'
        cv = []
        for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(X_tr, y_tr, groups=groups):
            m = M.build_model(kind, cw).fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
            cv.append(macro_f1(y_tr[va_idx], m.predict(X_tr.iloc[va_idx])))
        m = M.build_model(kind, cw).fit(X_tr, y_tr)
        pred_va = m.predict(X_va)
        results[name] = (kind, cw, np.mean(cv), macro_f1(y_va, pred_va), pred_va)
        print(f'{name:18s} GroupKFold {np.mean(cv):.4f} ± {np.std(cv):.4f} | Validation {results[name][3]:.4f}')

    # 모델 선택은 GroupKFold CV 평균 기준(Validation 하나에 과적합되는 것 방지)
    best = max(results, key=lambda k: results[k][2])
    kind, cw, cv_score, va_score, pred_va = results[best]
    print(f'\n선택: {best} (CV {cv_score:.4f}, Validation {va_score:.4f})')
    print(classification_report(y_va, pred_va, labels=M.LABELS, digits=4, zero_division=0))
    print('Confusion matrix (행=정답, 열=예측)\n', confusion_matrix(y_va, pred_va, labels=M.LABELS))

    if args.refit_train_val:
        final = M.build_model(kind, cw).fit(pd.concat([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    else:
        final = M.build_model(kind, cw).fit(X_tr, y_tr)
    joblib.dump({'model': final, 'features': list(X_tr.columns), 'config': best}, args.model_out)
    print('모델 저장:', args.model_out)


if __name__ == '__main__':
    main()
