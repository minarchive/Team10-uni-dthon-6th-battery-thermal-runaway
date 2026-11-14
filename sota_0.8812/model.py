"""모델 정의: 시계열 LightGBM 설정, 이미지 모델(EfficientNet-B0), 이미지 입력 구성, 인과적 단조 디코딩.

모든 입력은 같은 실험의 현재·과거 관측만 사용한다 (relative_time_s 기준).
"""
import re
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image

SEED = 42
LABELS = np.array([1, 2, 3, 4, 5, 6])
# 과충전(tr_method 2) 실험용 모델에서 제외하는 '압력 절대값' 피처 (시작 압력이 실험마다 크게 달라 일반화를 해침)
P_ABS = r'^meas_p($|_cmax$|_m60$|_w\d+_(mean|min|max)$)'
LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=16, verbose=-1)
IMG_SIZE = 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seed_everything(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def cols_without_abs_pressure(cols):
    """과충전 실험용 모델 피처: 압력 절대값 + 과충전 배율(v_peak_ratio) 제외 (5-fold: 각형 과충전 0.785 → 0.811)"""
    return [c for c in cols if not re.search(P_ABS + r'|^v_peak_ratio$', c)]


# ---------------------------------------------------------------- 이미지
class VisionNet(nn.Module):
    """EfficientNet-B0 (ImageNet 사전학습) + 6-class 헤드."""
    def __init__(self, pretrained=True):
        super().__init__()
        m = tvm.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        d = m.classifier[1].in_features; m.classifier = nn.Identity()
        self.net, self.head = m, nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 6))

    def forward(self, x):
        return self.head(self.net(x))


def load_images(df, image_root):
    """df.image_path 순서대로 256x256 RGB uint8 배열"""
    from concurrent.futures import ThreadPoolExecutor
    def ld(p):
        with Image.open(Path(image_root) / p) as im:
            return np.asarray(im.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), dtype=np.uint8)
    with ThreadPoolExecutor(16) as ex:
        return np.stack(list(ex.map(ld, df.image_path.astype(str).tolist())))


def past_index(df, lag):
    """같은 실험에서 t-lag초 이전 가장 최근 프레임의 행 위치 (없으면 첫 프레임). df 행 순서 기준."""
    out = np.arange(len(df)); t = df.relative_time_s.values
    for _, idx in df.groupby('experiment_id').indices.items():
        idx = idx[np.argsort(t[idx], kind='stable')]; tt = t[idx]
        out[idx] = idx[np.maximum(np.searchsorted(tt, tt - lag, side='right') - 1, 0)]
    return out


def past_index_grid(df, lag, s):
    """학습 증강: 실험을 stride s(5/10/15초 간격) 격자로 볼 때의 t-lag 이전 프레임 (Test 다운샘플 재현)"""
    out = np.arange(len(df)); t = df.relative_time_s.values
    for _, idx in df.groupby('experiment_id').indices.items():
        idx = idx[np.argsort(t[idx], kind='stable')]
        for o in range(s):
            q = idx[o::s]; tq = t[q]; out[q] = q[np.maximum(np.searchsorted(tq, tq - lag, side='right') - 1, 0)]
    return out


def first_index(df):
    out = np.arange(len(df)); t = df.relative_time_s.values
    for _, idx in df.groupby('experiment_id').indices.items(): out[idx] = idx[np.argmin(t[idx])]
    return out


def vision_input(I, rows, f0, p15, p60, dev, flip=None):
    """실험 기준 입력 3채널: [현재-첫 프레임, 현재-15초 전, 현재-60초 전] (흑백)"""
    g = lambda r: torch.from_numpy(I[r]).to(dev).permute(0, 3, 1, 2).float().mean(1, keepdim=True) / 255.
    cur = g(rows); x = torch.cat([cur - g(f0), cur - g(p15), cur - g(p60)], 1) * 2 + 0.5
    if flip is not None: x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    return (x - MEAN.to(dev)) / STD.to(dev)


# ---------------------------------------------------------------- 디코딩
def viterbi(df, P):
    """인과적 단조 디코딩: 각 시점까지의 확률로 '단계가 내려가지 않는' 경로의 MAP를 구하고 마지막 상태를 출력 (하이퍼파라미터 없음)."""
    pred = np.zeros(len(df), int); t = df.relative_time_s.values
    for _, idx in df.reset_index(drop=True).groupby('experiment_id').indices.items():
        idx = idx[np.argsort(t[idx], kind='stable')]; L = np.log(np.clip(P[idx], 1e-6, 1)); best = None
        for j, i in enumerate(idx):
            best = L[j] if best is None else np.maximum.accumulate(best) + L[j]; pred[i] = best.argmax() + 1
    return pred
