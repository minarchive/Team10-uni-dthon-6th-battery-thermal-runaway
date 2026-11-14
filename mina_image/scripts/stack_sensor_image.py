"""센서 모델에 이미지 확률을 더하면 얼마나 좋아지는가 (같은 data/folds.parquet fold).

입력: outputs/preds/image_probs.parquet (build_image_features.py), 센서 CSV(data_link)
비교(GBM, 5-fold, 실험 단위): 조건+센서+과거 시간 feature  vs  + 이미지 확률(현재)  vs  + 이미지 확률의 과거 EMA/기대 단계
주의: 이미지 확률은 같은 fold 기준 OOF 라서 2단계(stacking) 모델의 검증 fold 정보가 1단계 학습에 섞여 있을 수 있다(보통 작지만 0은 아님).
      최종 제출용 평가는 중첩 CV 또는 Train->Val 분할로 한 번 더 확인할 것.
사용: python scripts/stack_sensor_image.py --model ft_dino_r50   (여러 개면 --model a b)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from common import DATA, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--model", nargs="+", required=True)
ap.add_argument("--probs", default="image_probs")
a = ap.parse_args()

img = pd.read_parquet(ROOT / "outputs" / "preds" / f"{a.probs}.parquet")
sens = pd.concat([pd.read_csv(DATA / s / f"{s}.csv") for s in ["train", "validation"]], ignore_index=True)
df = img[img.label >= 1].merge(sens.drop(columns=["experiment_id", "relative_time_s", "label"]), on="sample_id", how="left")
df = df.sort_values(["experiment_id", "relative_time_s"], kind="stable").reset_index(drop=True)
df["t1_plus_t2"] = df.meas_t1 + df.meas_t2
SENS = [c for c in ["meas_v", "meas_t1", "meas_t2", "meas_p", "meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf",
                    "meas_n2o", "meas_no", "meas_no2", "meas_so2", "atmp_t", "thermal_t", "indu_t", "indu_t.1", "t1_plus_t2"]
        if df[c].nunique(dropna=True) > 1]
COND = ["bat_class", "bat_cc", "soc_status", "tr_method"]
KEY = [c for c in ["meas_t1", "meas_t2", "thermal_t", "meas_v", "meas_p", "meas_co2", "meas_co", "atmp_t", "t1_plus_t2"] if c in SENS]
IMGC = {m: [f"{m}_p{k}" for k in range(1, 7)] for m in a.model}


def temporal(g):
    t = g.relative_time_s.values.astype(float)
    out = {}
    for c in KEY:
        v = g[c].values.astype(float)
        for lag in (30, 120):
            j = np.maximum(np.searchsorted(t, t - lag, side="right") - 1, 0)
            out[f"{c}_d{lag}"] = v - v[j]
        out[f"{c}_cummax"] = np.fmax.accumulate(np.where(np.isnan(v), -np.inf, v))
    out["elapsed"] = t
    for m, cols in IMGC.items():
        P = g[cols].values.astype(float)
        e = P @ np.arange(1, 7)
        out[f"{m}_exp"] = e
        for tau in (30, 120):
            ema = np.empty_like(P)
            ema[0] = P[0]
            for i in range(1, len(P)):
                al = np.exp(-(t[i] - t[i - 1]) / tau)
                ema[i] = al * ema[i - 1] + (1 - al) * P[i]
            for k in range(6):
                out[f"{m}_ema{tau}_p{k + 1}"] = ema[:, k]
            out[f"{m}_ema{tau}_exp"] = ema @ np.arange(1, 7)
        j = np.maximum(np.searchsorted(t, t - 30, side="right") - 1, 0)
        out[f"{m}_exp_d30"] = e - e[j]
    return pd.DataFrame(out, index=g.index)


T = pd.concat([temporal(g) for _, g in df.groupby("experiment_id", sort=False)]).replace(-np.inf, np.nan)
df = pd.concat([df, T], axis=1)
sensor_t = [c for c in T.columns if not any(c.startswith(m) for m in a.model)]
img_now = [c for cols in IMGC.values() for c in cols] + [f"{m}_exp" for m in a.model]
img_t = [c for c in T.columns if any(c.startswith(m) for m in a.model) and c not in img_now]
L = list(range(1, 7))


def cv(cols, name):
    f1s, oof = [], np.zeros(len(df), int)
    for f in range(5):
        tr, va = (df.fold != f).values, (df.fold == f).values
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=5, class_weight="balanced", random_state=0)
        m.fit(df.loc[tr, cols], df.label[tr])
        oof[va] = m.predict(df.loc[va, cols])
        f1s.append(f1_score(df.label[va], oof[va], average="macro", labels=L, zero_division=0))
    pc = f1_score(df.label, oof, average=None, labels=L, zero_division=0)
    print(f"{name:34s} F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | OOF {f1_score(df.label, oof, average='macro', labels=L, zero_division=0):.4f}"
          f" | adj {float((np.abs(oof - df.label.values) <= 1).mean()):.4f} | per-class {[round(v, 2) for v in pc]}", flush=True)
    return oof, f1s


base = COND + SENS + sensor_t
res = {}
res["sensor"] = cv(base, "센서+과거 시간 feature")
res["img_only_now"] = cv(img_now, "이미지 확률만(현재)")
res["sensor+img_now"] = cv(base + img_now, "센서 + 이미지 확률(현재)")
res["sensor+img_all"] = cv(base + img_now + img_t, "센서 + 이미지 확률(현재+과거 EMA)")
d = np.array(res["sensor+img_all"][1]) - np.array(res["sensor"][1])
print("\nfold별 차이(센서+이미지 전체 - 센서):", np.round(d, 4), "평균", round(float(d.mean()), 4), "| 이미지가 이긴 fold 수", int((d > 0).sum()), "/ 5")
yt, s, c = df.label.values, res["sensor"][0], res["sensor+img_all"][0]
print(f"센서 오류율 {np.mean(s != yt):.3f} -> 이미지 추가 {np.mean(c != yt):.3f} | 고쳐진 {np.mean((s != yt) & (c == yt)):.3f} 새로 틀린 {np.mean((s == yt) & (c != yt)):.3f}")
for k in L:
    mk = yt == k
    print(f"L{k}: 오류율 {np.mean(s[mk] != k):.3f} -> {np.mean(c[mk] != k):.3f} (n={mk.sum()})")
