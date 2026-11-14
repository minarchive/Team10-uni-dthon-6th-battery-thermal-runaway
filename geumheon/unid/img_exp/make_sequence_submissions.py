"""Write Test predictions from saved Train-fold HSMM/GRU outputs without retraining."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402


def write(ids, pred, order, path):
    s = pd.Series(np.asarray(pred, dtype=int), index=pd.Index(ids.astype(str), name='sample_id'))
    if not s.index.is_unique or set(order) != set(s.index):
        raise ValueError('Saved predictions do not exactly match sample_submission IDs')
    sub = pd.DataFrame({'sample_id': order.values, 'label': s.loc[order.values].values})
    if not sub.label.isin(F.LABELS).all():
        raise ValueError('Prediction contains an invalid label')
    sub.to_csv(path, index=False)
    print(f'{path}: {len(sub)} rows; labels={sub.label.value_counts().sort_index().to_dict()}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, default=Path('seq_predictions.npz'))
    ap.add_argument('--out_dir', type=Path, default=Path('.'))
    args = ap.parse_args()
    z = np.load(args.predictions)
    ids = z['test_sample_id'].astype(str)
    test = pd.read_csv(args.data_dir / 'test' / 'test.csv', dtype={'sample_id': 'string'})
    if not np.array_equal(ids, test.sample_id.astype(str).to_numpy()):
        raise ValueError('Saved test probability order does not match test.csv')
    order = pd.read_csv(args.data_dir / 'sample_submission.csv', dtype={'sample_id': 'string'}).sample_id
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # HSMM probabilities are already causally filtered; argmax is the decoder.
    p_hsmm = z['test_hsmm']
    write(ids, F.LABELS[p_hsmm.argmax(1)], order, args.out_dir / 'sub_ts_hsmm_traincv.csv')

    # Match the sequence model's evaluated post-processing rule.
    p_gru = z['test_gru']
    pred_gru = F.postprocess(test.reset_index(drop=True), p_gru, 0.7)
    write(ids, pred_gru, order, args.out_dir / 'sub_ts_gru_traincv.csv')


if __name__ == '__main__':
    main()
