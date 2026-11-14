"""센서(시계열)와 이미지의 관계 EDA + '센서 모델에 이미지가 무엇을 더하는가' 근사 실험.

1) 센서 feature 와 단계(label)의 상관, 센서 간 상관
2) 이미지 통계(색 비율 등)와 센서(thermal_t, meas_t1/t2 ...)의 상관
3) 같은 5-fold(folds.parquet 과 동일 절차)로 GBM 비교: 조건만 / 센서(현재) / 센서+시간 / 센서+과거 시간 feature / +이미지 통계 / 이미지 통계만
   (이미지 통계는 임베딩이 아니라 거친 대리 변수이므로 '이미지가 추가로 주는 정보의 하한 근사'로만 해석)
미래 정보 금지: 모든 시간 feature 는 같은 실험의 과거 행만 사용(lag 차이, cummax, EMA).
사용: python scripts/eda_sensor_image.py --data <battery_data_light> --out outputs/figures [--image_stride 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="outputs/figures")
ap.add_argument("--image_stride", type=int, default=4, help="실험별 N 프레임마다 1장씩 이미지 통계 계산(0=이미지 생략)")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
data, out = Path(a.data), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

frames = []
for s in ["train", "validation"]:
    d = pd.read_csv(data / s / f"{s}.csv")
    d["split"] = s
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
# make_folds.py 와 동일한 분할(같은 순서·같은 시드)
df["fold"] = -1
for f, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=a.seed).split(df, df.label, df.experiment_id)):
    df.loc[df.index[va], "fold"] = f
df = df.sort_values(["experiment_id", "relative_time_s"], kind="stable").reset_index(drop=True)

SENS = ["meas_v", "meas_t1", "meas_t2", "meas_p", "meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf",
        "meas_n2o", "meas_no", "meas_no2", "meas_so2", "atmp_t", "thermal_t", "indu_t", "indu_t.1"]
COND = ["bat_class", "bat_cc", "soc_status", "tr_method"]
df["t1_plus_t2"] = df.meas_t1 + df.meas_t2
SENS.append("t1_plus_t2")
const = [c for c in SENS if df[c].nunique(dropna=True) <= 1]
SENS = [c for c in SENS if c not in const]
print("상수 컬럼 제외:", const, "| 결측률:", df[SENS].isna().mean()[lambda s: s > 0].round(3).to_dict())

# ---- 1) 상관
y = df.label
rows = []
for c in SENS:
    ok = df[c].notna()
    rho = spearmanr(df.loc[ok, c], y[ok]).correlation
    # 실험 평균 제거 후 상관(실험 내 변화와 단계의 관계)
    z = df.loc[ok, c] - df.loc[ok].groupby("experiment_id")[c].transform("mean")
    rho_w = spearmanr(z, y[ok]).correlation
    rows.append(dict(feature=c, spearman_with_label=rho, spearman_within_experiment=rho_w))
corr = pd.DataFrame(rows).sort_values("spearman_with_label", key=abs, ascending=False)
corr["spearman_with_label"] = corr.spearman_with_label.round(3)
corr["spearman_within_experiment"] = corr.spearman_within_experiment.round(3)
print("\n== 센서 vs 단계 (Spearman, 전체 / 실험 평균 제거 후)")
print(corr.to_string(index=False))
corr.to_csv(out / "sensor_label_corr.csv", index=False)
sc = df[SENS].corr(method="spearman")
hi = [(i, j, round(sc.loc[i, j], 2)) for k, i in enumerate(SENS) for j in SENS[k + 1:] if abs(sc.loc[i, j]) > 0.9]
print("\n센서 간 |Spearman|>0.9 쌍(중복 정보):", hi)

# ---- 시간 기반 과거 feature (causal)
KEY = [c for c in ["meas_t1", "meas_t2", "thermal_t", "meas_v", "meas_p", "meas_co2", "meas_co", "atmp_t", "t1_plus_t2"] if c in SENS]


def add_temporal(g):
    t = g.relative_time_s.values.astype(float)
    res = {}
    for c in KEY:
        v = g[c].values.astype(float)
        for lag in (30, 120):
            j = np.maximum(np.searchsorted(t, t - lag, side="right") - 1, 0)
            res[f"{c}_d{lag}"] = v - v[j]
        res[f"{c}_cummax"] = np.fmax.accumulate(np.where(np.isnan(v), -np.inf, v))
        res[f"{c}_over_start"] = v - v[0]
    res["elapsed"] = t
    return pd.DataFrame(res, index=g.index)


T = pd.concat([add_temporal(g) for _, g in df.groupby("experiment_id", sort=False)])
T = T.replace(-np.inf, np.nan)
df = pd.concat([df, T], axis=1)
TFEAT = list(T.columns)

# ---- 이미지 통계 (거친 대리 변수)
IMG = []
if a.image_stride > 0:
    cache = out / "image_stats.parquet"
    df["_k"] = df.groupby("experiment_id").cumcount()
    sel = df[df._k % a.image_stride == 0]
    if cache.exists():
        st = pd.read_parquet(cache)
    else:
        recs = []
        for i, r in enumerate(sel.itertuples()):
            x = np.asarray(Image.open(data / r.split / r.image_path).convert("RGB")).astype(np.int16)
            R, G, B = x[..., 0], x[..., 1], x[..., 2]
            recs.append(dict(sample_id=r.sample_id,
                             img_magenta=((R > 150) & (B > 150) & (G < 90)).mean(), img_blue=((B > 150) & (R < 90)).mean(),
                             img_green=((G > 150) & (R < 150) & (B < 150)).mean(), img_yellow=((R > 180) & (G > 180) & (B < 120)).mean(),
                             img_red=((R > 180) & (G < 90) & (B < 90)).mean(), img_white=((R > 200) & (G > 200) & (B > 180)).mean(),
                             img_R=R.mean(), img_G=G.mean(), img_B=B.mean(), img_Gstd=G.std()))
            if i % 2000 == 0:
                print("image stats", i, "/", len(sel), flush=True)
        st = pd.DataFrame(recs)
        st.to_parquet(cache, index=False)
    df = df.merge(st, on="sample_id", how="left")
    IMG = [c for c in st.columns if c != "sample_id"]
    d = df.dropna(subset=IMG)
    print(f"\n이미지 통계 표본 {len(d)}행")
    ci = []
    for ic in IMG:
        for sc_ in ["thermal_t", "meas_t1", "meas_t2", "meas_v", "meas_co2"]:
            if sc_ in SENS:
                ok = d[sc_].notna()
                ci.append(dict(image=ic, sensor=sc_, spearman=round(spearmanr(d.loc[ok, ic], d.loc[ok, sc_]).correlation, 3)))
    cim = pd.DataFrame(ci).pivot(index="image", columns="sensor", values="spearman")
    print("\n== 이미지 통계 vs 센서 (Spearman)")
    print(cim.to_string())
    cim.to_csv(out / "image_sensor_corr.csv")
    lab_c = pd.Series({ic: round(spearmanr(d[ic], d.label).correlation, 3) for ic in IMG})
    print("\n이미지 통계 vs 단계:", lab_c.to_dict())
    df = d.reset_index(drop=True)

# ---- 3) 모델 비교 (같은 fold)
L = list(range(1, 7))


def cv(cols, name):
    f1s, oof = [], np.zeros(len(df), int)
    for f in range(5):
        tr, va = df.fold != f, df.fold == f
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=5, class_weight="balanced", random_state=0)
        m.fit(df.loc[tr, cols], df.label[tr])
        oof[va.values] = m.predict(df.loc[va, cols])
        f1s.append(f1_score(df.label[va], oof[va.values], average="macro", labels=L, zero_division=0))
    pc = f1_score(df.label, oof, average=None, labels=L, zero_division=0)
    adj = float((np.abs(oof - df.label.values) <= 1).mean())
    r = dict(model=name, n_feat=len(cols), f1_mean=np.mean(f1s), f1_std=np.std(f1s), oof_f1=f1_score(df.label, oof, average="macro", labels=L, zero_division=0),
             adj_acc=adj, per_class=[round(v, 2) for v in pc])
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return r, oof


print(f"\n== 5-fold GBM 비교 ({len(df)}행, 이미지 통계가 있는 행만)" if IMG else f"\n== 5-fold GBM 비교 ({len(df)}행)")
res, oofs = [], {}
variants = {
    "조건만(bat_class 등)": COND,
    "센서(현재 시점)": COND + SENS,
    "센서+경과시간": COND + SENS + ["elapsed"],
    "센서+과거 시간 feature": COND + SENS + TFEAT,
}
if IMG:
    variants["이미지 통계만"] = IMG
    variants["센서+과거+이미지 통계"] = COND + SENS + TFEAT + IMG
for name, cols in variants.items():
    r, oofs[name] = cv(cols, name)
    res.append(r)
pd.DataFrame(res).to_csv(out / "sensor_image_gbm.csv", index=False)

if IMG:
    s, c = oofs["센서+과거 시간 feature"], oofs["센서+과거+이미지 통계"]
    yt = df.label.values
    print("\n== 상보성: 센서+과거 모델이 틀린 프레임 중 이미지 통계를 더하면 맞게 되는 비율 / 반대")
    print(f"센서 오류율 {np.mean(s != yt):.3f} | 이미지 추가 후 {np.mean(c != yt):.3f} | 고쳐진 {np.mean((s != yt) & (c == yt)):.3f} | 새로 틀린 {np.mean((s == yt) & (c != yt)):.3f}")
    print("\n단계별 오류율 (센서+과거 -> +이미지):")
    for k in L:
        m = yt == k
        print(f"L{k}: {np.mean(s[m] != k):.3f} -> {np.mean(c[m] != k):.3f}  (n={m.sum()})")
