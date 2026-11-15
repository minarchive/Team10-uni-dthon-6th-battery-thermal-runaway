"""oc4 R_J6wide 센서 Test 확률(final_oc4_gpu/test_probs_R_J6wide.npy) + v3_sota_et0.8920 EfficientNet-B0(차이 영상, seed 42·7) Test 확률.
CV/Val 없이 Test 제출 파일만 만든다. 비중 0.1, 후처리 τ 0.7 (oc4 와 동일)."""
import numpy as np, pandas as pd, torch
import effnet_diff_probs as E                    # 스냅샷 model 로딩·predict 재사용
from final_oc import F, load, D

E.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
te = load(D, 'test').reset_index(drop=True); te['sample_id'] = te.sample_id.astype(str)
img = E.SM.load_images(te, D / 'test'); nets = []
for s in (42, 7):
    n = E.SM.VisionNet(pretrained=False).to(E.dev)
    n.load_state_dict(torch.load(E.SNAP / 'weights' / f'vision_seed{s}.pt', map_location=E.dev)); nets.append(n)
Pv = E.predict(nets, img, te).astype(np.float32)
np.savez(E.HERE / 'eff_diff_b0_trval.npz', sample_id=te.sample_id.values, probs=Pv)
pS = np.load(E.HERE / 'final_oc4_gpu' / 'test_probs_R_J6wide.npy')
lab = F.postprocess(te, 0.9 * pS + 0.1 * Pv, 0.7)
ss = pd.read_csv(D / 'sample_submission.csv', dtype={'sample_id': str})
sub = ss[['sample_id']].merge(pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab}), on='sample_id', how='left')
assert sub.label.notna().all() and len(sub) == len(ss)
out = E.HERE / 'final_oc4_eff' / 'submission_R_J6wide_effdiff0.1.csv'; out.parent.mkdir(exist_ok=True)
sub.label = sub.label.astype(int); sub.to_csv(out, index=False)
print('device', E.dev, '| wrote', out, sub.label.value_counts().sort_index().to_dict())
