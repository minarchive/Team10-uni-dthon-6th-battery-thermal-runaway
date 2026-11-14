"""시계열 전용 모델 추론.

예) python test_ts.py --input ./data/battery_data_light/test/test.csv \
                      --model ./model_ts.joblib --output ./submission_ts.csv \
                      --sample_submission ./data/battery_data_light/sample_submission.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import ts_model as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('model_ts.joblib'))
    p.add_argument('--output', type=Path, default=Path('submission_ts.csv'))
    p.add_argument('--sample_submission', type=Path, default=None)
    args = p.parse_args()

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique
    bundle = joblib.load(args.model)
    X = M.build_features(frame)[bundle['features']]
    if 'tau' in bundle:   # v3: 확신 cummax 후처리 (실험별 시간순, 과거 예측만 사용)
        from exp_post import _per_exp, conf_cummax
        P = bundle['model'].predict_proba(X)
        labels = _per_exp(frame, P, lambda P, t: conf_cummax(P, bundle['tau']))
    else:
        labels = bundle['model'].predict(X)
    pred = pd.Series(labels.astype(int), index=frame.sample_id.values)

    if args.sample_submission is not None:
        order = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id']
        assert set(order) == set(pred.index), '입력 CSV와 제출 양식의 ID가 다릅니다.'
        ids = order.tolist()
    else:
        ids = frame.sample_id.tolist()
    sub = pd.DataFrame({'sample_id': ids, 'label': pred.loc[ids].values})
    assert sub.label.isin(M.LABELS).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(sub)}행)')


if __name__ == '__main__':
    main()
