"""experiment1 공통 모듈: 데이터 로딩, 이미지 통계 피처, 인과(과거 방향) 시계열 피처.

규칙 준수: 모든 시계열 피처는 같은 experiment_id 안에서 relative_time_s <= 현재 시점 관측만 사용한다.
(실험 전체 통계, 총 길이, 종료 시각, 미래 관측 기반 보간/이동창은 사용하지 않는다.)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", "/home/ubuntu/data/battery_data_light"))
CACHE = ROOT / "cache"
OUT = ROOT / "outputs"
SEED = 42
LABELS = list(range(1, 7))

GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
SENSOR_BASE = ["meas_v", "meas_t1", "meas_t2", "meas_p", "atmp_t", "thermal_t"] + GAS
COND = ["bat_class", "bat_cc", "soc_status", "tr_method"]
# 이미지 크기는 bat_class 지름길 -> 별도 그룹으로 분리해 ablation
SIZE_FEATS = ["size_w", "size_h", "size_aspect", "size_npix"]


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------
def load_split(split: str, data_dir: Path | None = None) -> pd.DataFrame:
    d = Path(data_dir or DATA_DIR)
    df = pd.read_csv(d / split / f"{split}.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    df["split"] = split
    df["img_file"] = df["image_path"].map(lambda p: str(d / split / p))
    return df


def load_all(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    return {s: load_split(s, data_dir) for s in ["train", "validation", "test"]}


# ---------------------------------------------------------------------------
# 이미지 피처 (이미지 1장만 사용 -> 다른 행 정보 사용 없음)
#   의사색 팔레트: 자홍(저온) -> 파랑 -> 청록 -> 초록 -> 노랑 -> 빨강 -> 흰색(고온)
#   hue 를 0~1 '상대 온도 인덱스'로 근사 변환 (원해상도에서 변환)
# ---------------------------------------------------------------------------
def _temp_index(a: np.ndarray):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    d = mx - mn
    hue = np.zeros_like(mx)
    m = d > 1e-6
    rc = (mx == r) & m
    gc = (mx == g) & m & ~rc
    bc = m & ~rc & ~gc
    hue[rc] = (((g - b)[rc] / d[rc]) % 6) * 60
    hue[gc] = ((b - r)[gc] / d[gc] + 2) * 60
    hue[bc] = ((r - g)[bc] / d[bc] + 4) * 60
    hue = np.where(hue > 330, 0.0, hue)                # 빨강 쪽으로 wrap
    idx = np.clip(300.0 - hue, 0, 300) / 300.0 * 0.833  # 300°(자홍)=0 ... 0°(빨강)=0.833
    sat = d / (mx + 1e-6)
    white = (sat < 0.25) & (mx > 0.85)
    idx = np.where(white, 1.0, idx)
    valid = ~((mx < 0.12) & ~white)                     # 거의 검은 픽셀 제외
    return idx, white, valid


def image_stats(path: str) -> dict:
    with Image.open(path) as im:
        w, h = im.size
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    idx, white, valid = _temp_index(a)
    v = idx[valid] if valid.any() else idx.ravel()
    q = np.percentile(v, [5, 25, 50, 75, 90, 95, 99])
    hist = np.histogram(v, bins=8, range=(0, 1.0001))[0] / len(v)
    gy = np.abs(np.diff(idx, axis=0)).mean() if h > 1 else 0.0
    gx = np.abs(np.diff(idx, axis=1)).mean() if w > 1 else 0.0
    hh, ww = idx.shape
    center = idx[hh // 4: 3 * hh // 4, ww // 4: 3 * ww // 4]
    out = {
        "size_w": w, "size_h": h, "size_aspect": w / h, "size_npix": w * h,
        "img_mean": float(v.mean()), "img_std": float(v.std()), "img_max": float(v.max()),
        "img_r": float(a[..., 0].mean()), "img_g": float(a[..., 1].mean()), "img_b": float(a[..., 2].mean()),
        "img_white": float(white.mean()), "img_hot67": float((v > 0.67).mean()), "img_hot83": float((v >= 0.83).mean()),
        "img_cold20": float((v < 0.2).mean()), "img_edge": float(gx + gy),
        "img_center_mean": float(center.mean()), "img_center_minus_all": float(center.mean() - idx.mean()),
    }
    for k, name in zip([5, 25, 50, 75, 90, 95, 99], q):
        out[f"img_p{k}"] = float(name)
    for i, hv in enumerate(hist):
        out[f"img_h{i}"] = float(hv)
    return out


def image_features(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(image_stats)(p) for p in df["img_file"])
    return pd.DataFrame(rows, index=df["sample_id"].values)


def load_image_cache(split: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"img_feats_{split}.parquet")


IMG_CORE = ["img_mean", "img_std", "img_max", "img_p50", "img_p90", "img_p99", "img_white", "img_hot67", "img_hot83", "img_edge", "img_center_mean"]


# ---------------------------------------------------------------------------
# 인과 시계열 피처
# ---------------------------------------------------------------------------
def derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t1_minus_t2"] = out["meas_t1"] - out["meas_t2"]
    out["t1_minus_amb"] = out["meas_t1"] - out["atmp_t"]
    out["thermal_minus_amb"] = out["thermal_t"] - out["atmp_t"]
    out["thermal_minus_t1"] = out["thermal_t"] - out["meas_t1"]
    out["indu_missing"] = out["indu_t"].isna().astype(int)
    out["gas_sum_log"] = np.log1p(out[GAS].sum(axis=1))
    return out


CAUSAL_SIGNALS = SENSOR_BASE + ["t1_minus_t2", "t1_minus_amb", "thermal_minus_amb", "gas_sum_log"]
DELTA_W = (15, 30, 60)
ROLL_W = (30, 60)


def causal_features(df: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """실험별로 따로 계산. df 는 sample_id/experiment_id/relative_time_s + signals 포함.
    생성: cummax, 첫 관측 대비 변화, cummax 대비 하락량, rolling mean/max(30,60s), 시간차 delta(15,30,60s)."""
    cols: dict[str, np.ndarray] = {}
    names = []
    for s in signals:
        names += [f"{s}__cummax", f"{s}__from_first", f"{s}__below_cummax"]
        names += [f"{s}__mean{w}" for w in ROLL_W] + [f"{s}__max{w}" for w in ROLL_W]
        names += [f"{s}__d{w}" for w in DELTA_W]
    out = pd.DataFrame(np.nan, index=df.index, columns=names)
    t_all = df["relative_time_s"].to_numpy(float)
    X_all = df[signals].to_numpy(float)
    grp = df.groupby("experiment_id", sort=False).indices
    pos = {n: i for i, n in enumerate(names)}
    buf = np.full((len(df), len(names)), np.nan)
    for _, ix in grp.items():
        ix = ix[np.argsort(t_all[ix], kind="mergesort")]
        t = t_all[ix]
        X = X_all[ix]
        tdx = pd.to_timedelta(t, unit="s")
        frame = pd.DataFrame(X, index=tdx, columns=signals)
        cmax = np.fmax.accumulate(X, axis=0)
        first = X[0:1]
        for j, s in enumerate(signals):
            buf[ix, pos[f"{s}__cummax"]] = cmax[:, j]
            buf[ix, pos[f"{s}__from_first"]] = X[:, j] - first[0, j]
            buf[ix, pos[f"{s}__below_cummax"]] = cmax[:, j] - X[:, j]
        for w in ROLL_W:
            r = frame.rolling(f"{w}s")  # (t-w, t] , 미래 관측 미사용
            m, mx = r.mean().to_numpy(), r.max().to_numpy()
            for j, s in enumerate(signals):
                buf[ix, pos[f"{s}__mean{w}"]] = m[:, j]
                buf[ix, pos[f"{s}__max{w}"]] = mx[:, j]
        for w in DELTA_W:
            lag = np.searchsorted(t, t - w, side="right") - 1  # t-w 이하의 마지막 관측
            ok = lag >= 0
            for j, s in enumerate(signals):
                d = np.full(len(t), np.nan)
                d[ok] = X[ok, j] - X[lag[ok], j]
                buf[ix, pos[f"{s}__d{w}"]] = d
    out[:] = buf
    return out


def build_feature_table(df: pd.DataFrame, img: pd.DataFrame) -> pd.DataFrame:
    """df: 한 개 이상의 split(또는 증강본). 인덱스 = 행 위치(0..n-1). img: sample_id 인덱스 이미지 피처."""
    d = derived(df.reset_index(drop=True))
    im = img.reindex(d["sample_id"].values)
    assert im.notna().all().all(), "이미지 피처 누락"
    im = im.reset_index(drop=True)
    d = pd.concat([d, im], axis=1)
    base_cols = ["sample_id", "experiment_id", "relative_time_s"]
    sig = CAUSAL_SIGNALS + IMG_CORE
    cf = causal_features(d[base_cols + sig], sig)
    keep = (["relative_time_s"] + COND + SENSOR_BASE + ["indu_t", "indu_missing", "t1_minus_t2", "t1_minus_amb",
            "thermal_minus_amb", "thermal_minus_t1", "gas_sum_log"] + SIZE_FEATS
            + [c for c in im.columns if c.startswith("img_")])
    keep = list(dict.fromkeys(keep))
    return pd.concat([d[keep], cf], axis=1)


def feature_groups(cols: list[str]) -> dict[str, list[str]]:
    g = {
        "time": ["relative_time_s"],
        "cond": COND,
        "size": SIZE_FEATS,
        "sensor_raw": [c for c in cols if c in SENSOR_BASE + ["indu_t", "indu_missing", "t1_minus_t2", "t1_minus_amb", "thermal_minus_amb", "thermal_minus_t1", "gas_sum_log"]],
        "img_raw": [c for c in cols if c.startswith("img_") and "__" not in c],
        "sensor_causal": [c for c in cols if "__" in c and not c.startswith("img_")],
        "img_causal": [c for c in cols if "__" in c and c.startswith("img_")],
    }
    return g


# ---------------------------------------------------------------------------
# 학습용 증강: 테스트는 5/10/15초 간격이 섞여 있으므로, 학습 실험에서 프레임을 랜덤하게 건너뛴 사본 생성
# (건너뛴 행만으로 인과 피처를 다시 계산하므로 미래 정보 누수 없음)
# ---------------------------------------------------------------------------
def subsample_gaps(df: pd.DataFrame, probs=(0.6, 0.25, 0.15), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep = []
    for _, ix in df.groupby("experiment_id", sort=False).indices.items():
        ix = ix[np.argsort(df["relative_time_s"].to_numpy()[ix], kind="mergesort")]
        p = 0
        while p < len(ix):
            keep.append(ix[p])
            p += rng.choice([1, 2, 3], p=probs)
    return df.iloc[np.array(keep)].copy()


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def causal_cummax_pred(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """라벨은 실험 내에서 단조 비감소 -> 과거 예측의 누적 최댓값(인과)."""
    out = np.array(pred).copy()
    t = df["relative_time_s"].to_numpy(float)
    for _, ix in df.groupby("experiment_id", sort=False).indices.items():
        ix = ix[np.argsort(t[ix], kind="mergesort")]
        out[ix] = np.maximum.accumulate(out[ix])
    return out


def causal_vote_pred(df: pd.DataFrame, pred: np.ndarray, k: int = 5, monotone: bool = True) -> np.ndarray:
    """최근 k개 관측의 예측 중앙값(과거만) 후 (선택) 누적 최댓값. 일시적 오탐 완화."""
    out = np.array(pred).copy()
    t = df["relative_time_s"].to_numpy(float)
    for _, ix in df.groupby("experiment_id", sort=False).indices.items():
        ix = ix[np.argsort(t[ix], kind="mergesort")]
        s = pd.Series(pred[ix]).rolling(k, min_periods=1).median().round().to_numpy()
        out[ix] = np.maximum.accumulate(s) if monotone else s
    return out.astype(int)
