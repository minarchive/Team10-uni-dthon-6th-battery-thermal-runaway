"""팀 LightGBM 파이프라인 로컬 재현 + 확장 실험 (CPU).
- 피처/후처리: mina_image/ts_model/features.py (GitHub main). 모델 설정·가스 정리·X1·과충전 규칙은 서버의 final_lgb.py(저장소에 없음, 대화 기록 기준)를 재구성.
- 폴드: mina_image/scripts/make_folds.py 와 같은 StratifiedGroupKFold(5, shuffle, 42) (이미지 크기 열만 생략).
산출물은 outputs/team/ 아래(데이터 파생물, 커밋 금지).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "mina_image" / "ts_model"))
import features as F  # noqa: E402

OUT = ROOT / "outputs" / "team"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/home/ubuntu/data/battery_data_light"))
L = list(range(1, 7))
GAS = list(F.GAS)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def f1m(y, p):
    return f1_score(y, p, labels=L, average="macro", zero_division=0)


# ----------------------------------------------------------------------------- 데이터/폴드
def load_table(seed=42):
    frames = []
    for s in ["train", "validation", "test"]:
        d = pd.read_csv(DATA_DIR / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"})
        d["split"] = s
        if "label" not in d:
            d["label"] = -1
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["sample_id"], df["experiment_id"] = df.sample_id.astype(str), df.experiment_id.astype(str)
    df["label"] = df["label"].astype(int)
    df.loc[df.split == "test", "label"] = -1
    df["fold"] = fold_ids(df, seed)
    return df


def fold_ids(df, seed=42, k=5):
    fold = np.full(len(df), -1)
    Lb = df[df.split != "test"]
    for f, (_, va) in enumerate(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(Lb, Lb.label, Lb.experiment_id)):
        fold[Lb.index[va]] = f
    return fold


# ----------------------------------------------------------------------------- 피처 (가스 정리 + X1)
def _cross(d, x, levels):
    out = {}
    eid = d.experiment_id.values
    idxs = pd.Series(np.arange(len(d))).groupby(eid).indices
    t_all = d.relative_time_s.values.astype(float)
    for lv in levels:
        o = np.full(len(d), np.nan)
        for e, ix in idxs.items():
            ix = ix[np.argsort(t_all[ix], kind="stable")]
            cm = np.maximum.accumulate(x[ix]); m = cm >= lv
            tf = t_all[ix][np.argmax(m)] if m.any() else np.inf
            o[ix] = np.where(t_all[ix] >= tf, t_all[ix] - tf, np.nan)
        out[lv] = o
    return out


def build_X(df, gas="sum"):
    """gas: 'sum' = 가스합만 meas_co2 열에 넣고 나머지 0 (final_lgb), 'all' = 원본 가스, 'none' = 가스 전부 0."""
    d = df.copy()
    s = d[GAS].sum(axis=1)
    if gas in ("sum", "none"):
        for g in GAS:
            d[g] = 0.0
        if gas == "sum":
            d["meas_co2"] = s
    base = F.build_features(d)
    base.index = np.arange(len(base))
    th = d.thermal_t.values.astype(float)
    eid = d.experiment_id.values
    thr = th - pd.Series(th).groupby(eid).transform("first").values
    X1 = {f"cx_thermal_{k}": v for k, v in _cross(d, th, [35, 40, 45, 50, 55, 60, 65, 70, 80, 100, 150, 200]).items()}
    X1.update({f"cx_thermal_rel_{k}": v for k, v in _cross(d, thr, [2, 5, 10, 15, 20, 30, 40]).items()})
    return pd.concat([base, pd.DataFrame(X1, index=base.index)], axis=1)


def get_features(df, tag, gas="sum"):
    p = CACHE / f"team_X_{tag}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    t0 = time.time(); X = build_X(df, gas); X.to_parquet(p); log(f"피처 생성 {tag}: {X.shape}, {time.time() - t0:.0f}s"); return X


# ----------------------------------------------------------------------------- 과충전 규칙 (final_lgb)
def tha_cummax(df):
    tha = (df.thermal_t - df.atmp_t).values.astype(float)
    out = np.zeros(len(df)); t = df.relative_time_s.values.astype(float)
    for _, ix in pd.Series(np.arange(len(df))).groupby(df.experiment_id.values).indices.items():
        ix = ix[np.argsort(t[ix], kind="stable")]; out[ix] = np.maximum.accumulate(tha[ix])
    return out


def pick_theta(tha_cm, y, rows_mask):
    v = tha_cm[rows_mask]; yy = (y[rows_mask] == 3).astype(int)
    grid = np.unique(np.nanpercentile(v, np.linspace(2, 98, 33))); best, bt = -1, grid[0]
    for g in grid:
        f = f1_score(yy, (v >= g).astype(int), zero_division=0)
        if f > best:
            best, bt = f, g
    return bt


def override(P, tha_rows, theta):
    P = P.copy(); l3 = tha_rows >= theta; p23 = P[:, 1] + P[:, 2]; sel = (P.argmax(1) >= 1) & (P.argmax(1) <= 2)
    P[sel, 1] = np.where(l3[sel], 0.0, p23[sel]); P[sel, 2] = np.where(l3[sel], p23[sel], 0.0)
    return P


def make_lgb(**kw):
    import lightgbm as lgb
    p = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
             reg_lambda=1.0, class_weight="balanced", n_jobs=-1, verbose=-1, random_state=42)
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def proba6(m, X):
    Z = np.zeros((len(X), 6)); Z[:, np.array(m.classes_) - 1] = m.predict_proba(X); return Z


def report(name, df_l, y, fold, P, base=None, tau=0.7, extra=""):
    """풀링 OOF F1(확신 cummax), fold 별 F1, 과충전/가열 F1, 베이스 대비 실험별 정확도 차 95% CI."""
    p = F.postprocess(df_l, P, tau); fs = np.array([f1m(y[fold == k], p[fold == k]) for k in range(5)]); fv = f1m(y, p)
    oc = (df_l.tr_method.values == 2)
    s = f"{name:44s} OOF {fv:.4f} | fold {np.round(fs, 3).tolist()} | 과충전 {f1m(y[oc], p[oc]):.4f} 가열 {f1m(y[~oc], p[~oc]):.4f} {extra}"
    ci = None
    if base is not None:
        e = df_l.experiment_id.values
        a = pd.Series(p == y).groupby(e).mean(); b = pd.Series(base == y).groupby(e).mean(); dl = (a - b).to_numpy()
        rng = np.random.default_rng(0); bs = [dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(2000)]
        ci = (dl.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((dl > 0).sum()), int((dl < 0).sum()))
        s += f" | 베이스 대비 Δ{fv - f1m(y, base):+.4f} 실험별정확도 Δ{ci[0]:+.4f} CI[{ci[1]:+.4f},{ci[2]:+.4f}] 개선{ci[3]}/악화{ci[4]}"
    log(s)
    return p, fv, ci
