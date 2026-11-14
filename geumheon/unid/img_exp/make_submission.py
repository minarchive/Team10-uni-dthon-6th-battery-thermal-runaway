"""시계열(v3 735) + CNN 확률 스태킹 → 분류기 체크포인트 + test 제출 CSV.

- Train 행의 CNN 확률: finetune.py 의 OOF (그 실험을 안 본 fold 모델)
- Val/Test 행의 CNN 확률: 5개 fold 모델 평균 (어느 모델도 Val/Test로 학습하지 않음)
- --train_val: Train+Val 로 LightGBM 학습 (Val 점수는 낼 수 없음)

    python make_submission.py --data_dir <battery_data_light> --probs ft_convnext_tiny.npz --out sub.csv [--train_val]
    # 저장한 모델 재사용 (학습 생략)
    python make_submission.py --data_dir <battery_data_light> --probs ft_convnext_tiny.npz --out sub.csv --model_in sub.model.joblib
"""
import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402
from exp_img import lgbm, load, log, score6  # noqa: E402
from exp_stack_img import prob_features  # noqa: E402


def make_X(frame, probs):
    return pd.concat([F.build_features(frame), prob_features(frame, probs)], axis=1).loc[frame.sample_id.values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--probs', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--train_val', action='store_true')
    ap.add_argument('--model_out', type=Path, default=None, help='학습 분류기 저장 경로 (기본: <out stem>.model.joblib)')
    ap.add_argument('--model_in', type=Path, default=None, help='저장한 분류기를 불러와 학습 없이 추론')
    args = ap.parse_args()

    if args.model_in and args.train_val:
        ap.error('--model_in과 --train_val은 함께 사용할 수 없습니다')

    zs = [np.load(p) for p in args.probs]
    probs = pd.DataFrame(np.mean([z['probs'] for z in zs], 0), index=zs[0]['sample_id'].astype(str))
    assert not probs.isna().any().any(), 'CNN 확률에 NaN 이 있습니다'
    tr, va, te = load(args.data_dir, 'train'), load(args.data_dir, 'validation'), load(args.data_dir, 'test')

    if args.model_in:
        artifact = joblib.load(args.model_in)
        m = artifact['model']
        columns = artifact['feature_columns']
        expected_probs = artifact.get('probability_files')
        given_probs = [p.name for p in args.probs]
        if expected_probs and expected_probs != given_probs:
            ap.error(f'모델 학습에 사용한 확률 파일은 {expected_probs} 입니다 (입력: {given_probs})')
        log(f'저장 모델 불러옴: {args.model_in} (학습 설정: {artifact.get("fit_mode", "unknown")})')
    else:
        if not args.train_val:
            fit_frame = tr
        else:
            fit_frame = pd.concat([tr, va], ignore_index=True)
        Xfit = make_X(fit_frame, probs)
        m = lgbm(42).fit(Xfit, fit_frame.label.astype(int))
        columns = list(Xfit.columns)
        model_out = args.model_out or args.out.with_name(f'{args.out.stem}.model.joblib')
        model_out.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            'model': m,
            'feature_columns': columns,
            'fit_mode': 'train_val' if args.train_val else 'train',
            'backend': os.environ.get('GBM_BACKEND', 'lgbm_cpu'),
            'probability_files': [str(p) for p in args.probs],
            'postprocess_tau': 0.7,
        }
        tmp = model_out.with_name(model_out.name + '.tmp')
        joblib.dump(artifact, tmp)
        tmp.replace(model_out)
        log(f'분류기 저장: {model_out}')
        if not args.train_val:
            vc = F.cap_rows(va).reset_index(drop=True)
            r = score6(vc, m.predict_proba(make_X(vc, probs).reindex(columns=columns)))
            log(f'VAL (Train 학습, 500행 캡) raw {r["raw"]:.4f}  post {r["post"]:.4f}  F1/cls {r["per_class"]}')
        log(f'학습: {"Train+Val" if args.train_val else "Train"} {len(fit_frame)}행')

    te = te.reset_index(drop=True)
    Xte = make_X(te, probs)
    if list(Xte.columns) != list(columns):
        raise ValueError('저장 모델의 입력 피처 목록이 현재 피처와 일치하지 않습니다')
    P = m.predict_proba(Xte)
    pred = F.postprocess(te, P, 0.7)
    s = pd.Series(pred.astype(int), index=te.sample_id.values)
    order = pd.read_csv(args.data_dir / 'sample_submission.csv', dtype={'sample_id': 'string'})['sample_id']
    assert set(order) == set(s.index)
    sub = pd.DataFrame({'sample_id': order.values, 'label': s.loc[order.values].values})
    assert sub.label.isin(F.LABELS).all() and len(sub) == len(order)
    sub.to_csv(args.out, index=False)
    log(f'저장: {args.out} ({len(sub)}행) 라벨 분포 {sub.label.value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    main()
