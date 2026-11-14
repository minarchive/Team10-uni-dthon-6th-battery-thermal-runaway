"""이미지 통계 피처를 split별로 계산해 cache/img_feats_{split}.parquet 로 저장 (이미 있으면 건너뜀, --force 로 재계산)."""
import argparse, time
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="디버그용: split당 앞 N장만")
a = ap.parse_args()
C.CACHE.mkdir(exist_ok=True, parents=True)
print("DATA_DIR =", C.DATA_DIR, flush=True)
for split in ["train", "validation", "test"]:
    f = C.CACHE / f"img_feats_{split}.parquet"
    if f.exists() and not a.force and not a.limit:
        print(f"[skip] {f.name} 존재", flush=True); continue
    df = C.load_split(split)
    if a.limit: df = df.head(a.limit)
    t0 = time.time()
    feats = C.image_features(df)
    print(f"[{split}] {len(feats)}장 -> {feats.shape[1]}개 피처, {time.time()-t0:.0f}s", flush=True)
    if not a.limit: feats.to_parquet(f)
