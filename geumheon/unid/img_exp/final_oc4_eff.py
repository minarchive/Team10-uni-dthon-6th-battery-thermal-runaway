"""oc4 R_J6wide 를 그대로 두고 이미지 모델만 ConvNeXt-tiny → v3_sota_et0.8920 EfficientNet-B0(차이 영상) 로 교체.

센서 모델은 final_oc4_gpu/ckpt 체크포인트를 그대로 재사용 (ckpt 를 심볼릭 링크로 공유, 새로 학습하지 않음).
이미지 확률: effnet_diff_probs.py 출력
  CV/Val : eff_diff_b0.npz        (fold 별 새로 학습한 EfficientNet, seed 42)
  Test   : eff_diff_b0_trval.npz  (스냅샷 가중치 seed 42·7 평균)
비중 W_IMG 0.1, 후처리 τ 0.7 은 oc4 와 동일.

  GBM_BACKEND=xgb_cuda python final_oc4_eff.py
"""
import os

import final_oc as O
import final_oc4 as Q4
from final_oc2 import cnn as _cnn

MAP = {'ft_convnext_tiny.npz': 'eff_diff_b0.npz', 'ft_convnext_tiny_trval.npz': 'eff_diff_b0_trval.npz'}


def cnn(path, ids):
    return _cnn(MAP[path], ids)


Q4.cnn = cnn
Q4.CFGS = {'R_J6wide': Q4.CFGS['R_J6wide']}
Q4.OUT = O.M.HERE / 'final_oc4_eff'
Q4.OUT.mkdir(exist_ok=True)
ck = Q4.OUT / 'ckpt'
if not ck.exists():
    os.symlink(O.M.HERE / 'final_oc4_gpu' / 'ckpt', ck)

if __name__ == '__main__':
    Q4.main()
