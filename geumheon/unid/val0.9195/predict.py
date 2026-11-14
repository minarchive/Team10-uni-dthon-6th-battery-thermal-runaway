"""추론 + 확신 cummax 후처리 → 제출 파일 생성. 라벨이 있으면 성능도 출력합니다.

Test 제출:
    python predict.py --input <데이터>/test/test.csv --output submission.csv \
                      --sample_submission <데이터>/sample_submission.csv

Validation 점수 재현 (0.9195):
    python predict.py --input <데이터>/validation/validation.csv --output val_pred.csv --cap500
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import features as F


def report(frame, pred):
    from sklearn.metrics import f1_score
    y = frame['label'].astype(int).to_numpy()
    print(f'Macro F1: {f1_score(y, pred, labels=F.LABELS, average="macro", zero_division=0):.4f}')
    names = {1: '1 원통형', 2: '2 파우치형', 3: '3 각형'}
    rows = {}
    for b in [1, 2, 3, '합계']:
        m = np.ones(len(y), bool) if b == '합계' else (frame['bat_class'].to_numpy() == b)
        s = dict(zip([f'L{c}' for c in F.LABELS],
                     f1_score(y[m], pred[m], labels=F.LABELS, average=None, zero_division=0) * 100))
        present = [c for c in F.LABELS if (y[m] == c).any()]
        s['Macro'] = f1_score(y[m], pred[m], labels=present, average='macro', zero_division=0) * 100
        rows[names.get(b, b)] = s
    print('\nbat_class × 라벨 F1 % (Macro는 해당 bat_class에 있는 라벨만 평균)')
    print(pd.DataFrame(rows).T.round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True, help='test.csv 등 입력 CSV')
    ap.add_argument('--model', type=Path, default=Path(__file__).with_name('model.joblib'))
    ap.add_argument('--output', type=Path, default=Path('submission.csv'))
    ap.add_argument('--sample_submission', type=Path, default=None, help='주면 제출 양식의 행 순서로 저장')
    ap.add_argument('--tau', type=float, default=None, help='후처리 확신 기준(기본: 모델에 저장된 0.7)')
    ap.add_argument('--no_postprocess', action='store_true', help='후처리 없이 argmax만 사용')
    ap.add_argument('--cap500', action='store_true',
                    help='평가용: 500행 초과 실험을 Test처럼 균등 다운샘플(Validation 점수 재현 시 사용)')
    args = ap.parse_args()

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique
    if args.cap500:
        frame = F.cap_rows(frame)
    frame = frame.reset_index(drop=True)

    bundle = joblib.load(args.model)
    X = F.build_features(frame)[bundle['features']]
    P = bundle['model'].predict_proba(X)
    tau = args.tau if args.tau is not None else bundle['tau']
    pred = F.LABELS[P.argmax(1)] if args.no_postprocess else F.postprocess(frame, P, tau)

    s = pd.Series(pred.astype(int), index=frame.sample_id.values)
    if args.sample_submission is not None:
        order = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id']
        assert set(order) == set(s.index), '입력 CSV와 제출 양식의 ID가 다릅니다.'
        ids = order.tolist()
    else:
        ids = frame.sample_id.tolist()
    sub = pd.DataFrame({'sample_id': ids, 'label': s.loc[ids].values})
    assert sub.label.isin(F.LABELS).all() and sub.label.notna().all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(sub)}행, 후처리={"없음" if args.no_postprocess else f"확신 cummax τ={tau}"})')

    if 'label' in frame.columns:
        report(frame, pred)


if __name__ == '__main__':
    main()
