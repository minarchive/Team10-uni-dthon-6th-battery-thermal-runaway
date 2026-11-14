"""이미지 캐시 변형: 패딩을 검정(0)으로(팔레트에 없는 값, 마스크 효과), 보간은 bilinear 로 확대 잡음 번짐 완화. 긴 변 224, 비율 유지, 가운데 배치, crop 없음. -> data/img224_padblack.npy"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from common import DATA, ROOT, table
df = table(); paths = df.img_path.values; S = 224
out = np.lib.format.open_memmap(ROOT / "data" / "img224_padblack.npy", mode="w+", dtype=np.uint8, shape=(len(paths), S, S, 3))
def put(i):
    im = Image.open(DATA / paths[i]).convert("RGB"); s = S / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BILINEAR)
    c = Image.new("RGB", (S, S), (0, 0, 0)); c.paste(im, ((S - im.size[0]) // 2, (S - im.size[1]) // 2)); out[i] = np.asarray(c)
with ThreadPoolExecutor(10) as ex:
    for k, _ in enumerate(ex.map(put, range(len(paths)))):
        if k % 10000 == 0: print(k, flush=True)
out.flush(); print("saved", out.shape)
