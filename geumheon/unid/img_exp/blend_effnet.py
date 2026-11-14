"""저장된 센서 Test 확률(Train+Val 5시드) + EfficientNet-B0(Train+Val) 0.1 결합 → 제출. 사용: python blend_effnet.py <out_dir> <cfg>"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '../val0.9195'); import features as F
out, nm = sys.argv[1], sys.argv[2]
D = '../data/battery_data_light'
te = pd.read_csv(f'{D}/test/test.csv', dtype={'sample_id': str})
c = np.load('ft_efficientnet_b0_trval.npz', allow_pickle=True)
pI = pd.DataFrame(c['probs'], index=c['sample_id'].astype(str)).loc[te.sample_id.values].to_numpy()
pS = np.load(f'{out}/test_probs_{nm}.npy')
lab = F.postprocess(te, 0.9 * pS + 0.1 * pI, 0.7)
ss = pd.read_csv(f'{D}/sample_submission.csv', dtype={'sample_id': str})
sub = ss[['sample_id']].merge(pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab}), on='sample_id', how='left')
assert sub.label.notna().all() and len(sub) == len(ss)
p = f'{out}/submission_{nm}_effnet0.1.csv'; sub.to_csv(p, index=False)
print('wrote', p, sub.label.value_counts().sort_index().to_dict())
