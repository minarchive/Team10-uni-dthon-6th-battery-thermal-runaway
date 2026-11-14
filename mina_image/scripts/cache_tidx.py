"""온도 인덱스 채널 입력 캐시: 원해상도에서 의사색 -> 상대 온도 인덱스(0~1, experiment1 의 hue 근사)로 변환한 뒤 긴 변 224 로 리사이즈 + 0 패딩.
1채널 값을 3채널로 복제해 data/img224_tidx.npy (N,224,224,3) uint8 로 저장. 이미지 1장만 사용(다른 행 정보 없음). 행 순서 = folds.parquet."""
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from common import DATA, ROOT, table
import importlib.util
_sp = importlib.util.spec_from_file_location("exp1_common", "/home/ubuntu/exp1_wt/experiment1/src/common.py")
C1 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C1)  # experiment1: _temp_index
df = table(); paths = df.img_path.values; S = 224
out = np.lib.format.open_memmap(ROOT / "data" / "img224_tidx.npy", mode="w+", dtype=np.uint8, shape=(len(paths), S, S, 3))


def put(i):
    a = np.asarray(Image.open(DATA / paths[i]).convert("RGB"), dtype=np.float32) / 255.0
    idx, white, valid = C1._temp_index(a)
    im = Image.fromarray(np.where(valid, idx, 0.0).astype(np.float32), mode="F")
    s = S / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
    arr = np.clip(np.asarray(im), 0, 1)
    c = np.zeros((S, S), np.float32)
    y0, x0 = (S - arr.shape[0]) // 2, (S - arr.shape[1]) // 2
    c[y0:y0 + arr.shape[0], x0:x0 + arr.shape[1]] = arr
    out[i] = np.repeat((c * 255).astype(np.uint8)[..., None], 3, axis=2)


with ThreadPoolExecutor(12) as ex:
    for k, _ in enumerate(ex.map(put, range(len(paths)))):
        if k % 5000 == 0: print(k, "/", len(paths), flush=True)
out.flush(); print("saved", out.shape)
