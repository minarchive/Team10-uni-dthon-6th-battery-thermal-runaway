"""joint MoE 출력(npz)의 Test 확률 → 제출 파일.

Test 확률은 학습 시 5개 fold 모델 평균으로 이미 저장되어 있음 (Test는 추론에만 사용).
후처리: 실험별 시간순 확신 cummax(τ=0.7), 학습·검증과 동일.

    python make_sub_joint.py joint_moe2_s_u4.npz sub_joint_moe2_s_u4.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402

DATA = Path('/home/nxcolelxu/unid/data/battery_data_light')
src, dst = Path(sys.argv[1]), Path(sys.argv[2])

z = np.load(src)
P = pd.DataFrame(z['probs'], index=z['sample_id'].astype(str))[z['split'] == 'test']
te = pd.read_csv(DATA / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
te['sample_id'] = te.sample_id.astype(str)
assert set(te.sample_id) == set(P.index), 'test sample_id mismatch'
assert np.isfinite(P.to_numpy()).all()
lab = F.postprocess(te, P.loc[te.sample_id].to_numpy(), 0.7)
pred = pd.Series(lab, index=te.sample_id)

ss = pd.read_csv(DATA / 'sample_submission.csv', dtype={'sample_id': 'string'})
ss['label'] = pred.loc[ss.sample_id.astype(str)].astype(int).to_numpy()
assert ss.label.between(1, 6).all() and len(ss) == len(te)
ss[['sample_id', 'label']].to_csv(dst, index=False)
print(f'wrote {dst}  rows {len(ss)}  label counts {ss.label.value_counts().sort_index().to_dict()}')

# 단조성 확인: 실험 안에서 라벨이 내려가지 않는지
te['label'] = pred.loc[te.sample_id].to_numpy()
bad = sum((np.diff(g.sort_values('relative_time_s').label.to_numpy()) < 0).any() for _, g in te.groupby('experiment_id'))
print(f'experiments with decreasing labels: {bad} / {te.experiment_id.nunique()}')
