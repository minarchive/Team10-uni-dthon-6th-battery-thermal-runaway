"""oc5 R_J6wide 와 같은 파이프라인에서 시계열 모델만 LightGBM 으로 학습 (이미지 = 기존 ConvNeXt-Tiny 확률 그대로).

- 이미지: ft_convnext_tiny.npz (CV/Val) · ft_convnext_tiny_trval.npz (Test) 캐시 확률을 그대로 사용, 재학습 없음, 비중 0.1
  (L2/L3 열화상 EDA: 2↔3 경계는 센서 AUC 0.90~0.95 vs 이미지 최고 0.74 → 이미지는 보조 신호로 유지, 시계열 쪽 용량을 키움)
- 시계열: 45°C 규칙 + 2~6단계 모델 + 시계 재설정, 과충전 증강 J6wide w 0.3 (oc4/oc5 와 동일), OOD 피처 미사용
- 체크포인트: 학습한 모델은 전부 ckpt/<cfg>/{cv_fold*_seed*, val_seed*, final_seed*}.joblib 로 저장,
  이미 있으면 다시 학습하지 않고 불러옴(중단 후 재개 가능)

트리 (LightGBM)
  L_J6wide_L31 : 잎 31, 트리 500, lr 0.03 (XGB 기본과 같은 크기, 비교용)
  L_J6wide_L63 : 잎 63, 트리 1500, lr 0.02 (기본 실행)
  L_J6wide_L127: 잎 127, 트리 2500, lr 0.015

  TAG=oc6_lgb63 OC6_CFGS=L_J6wide_L63 python final_oc6_lgb.py
"""
import os

os.environ.setdefault('TAG', 'oc6_lgb')
import final_oc as O  # noqa: E402
import final_oc4 as Q4  # noqa: E402
from lgb_wrap import WLGB  # noqa: E402

TREES = {
    '_L31':  dict(num_leaves=31, n_estimators=500, learning_rate=0.03),
    '_L63':  dict(num_leaves=63, n_estimators=1500, learning_rate=0.02),
    '_L127': dict(num_leaves=127, n_estimators=2500, learning_rate=0.015),
}
CFGS = {f'L_J6wide{k}': dict(aug='wide', w=0.3, reset=True, tree=k) for k in TREES}
CFGS = {k: CFGS[k] for k in os.environ.get('OC6_CFGS', 'L_J6wide_L63').split(',')}


def fit_ckpt(path, seed, X, y, w):
    """학습한 모델은 항상 저장 (원자적 교체). 있으면 불러옴. 트리 크기는 ckpt/<cfg>/ 이름으로 결정."""
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return joblib.load(path)
    m = WLGB(seed, **TREES[CFGS[path.parent.name]['tree']])
    m.fit(X, y, w)
    tmp = path.with_name(path.name + '.tmp')
    joblib.dump(m, tmp); tmp.replace(path)
    O.log(f'saved {path.relative_to(Q4.OUT)}')
    return m


Q4.fit_ckpt = fit_ckpt
Q4.CFGS = CFGS
Q4.OUT = O.M.HERE / f'final_oc6_{os.environ["TAG"]}'
Q4.OUT.mkdir(exist_ok=True)

if __name__ == '__main__':
    O.log(f'LightGBM cfgs {CFGS} | out {Q4.OUT.name}')
    Q4.main()
