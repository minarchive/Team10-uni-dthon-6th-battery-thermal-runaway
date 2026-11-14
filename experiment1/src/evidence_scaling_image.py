"""성능에 critical 한 전처리 요소의 직접 측정: (A) 피처 스케일링/상대화, (B) 이미지 가공 방식.
결과: outputs/evidence/*.csv, *.png  (데이터 파생물 -> 커밋 금지)
실행: DATA_DIR=... python -u src/evidence_scaling_image.py [--smoke]
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import common as C
import pipeline as P_

log = P_.log
ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
OUT = C.OUT / "evidence"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

lab0 = pd.concat([C.load_split(s) for s in ("train", "validation")]).sort_values(["experiment_id", "relative_time_s"], kind="mergesort").reset_index(drop=True)
if a.smoke:
    ids = lab0.experiment_id.drop_duplicates().head(10)
    lab0 = lab0[lab0.experiment_id.isin(ids)].reset_index(drop=True)
img_cache = pd.concat([C.load_image_cache(s) for s in ("train", "validation", "test")])
X = C.build_feature_table(lab0, img_cache)
y = lab0.label.to_numpy(); g = lab0.experiment_id.to_numpy()
grp = C.feature_groups(list(X.columns))
S2 = list(dict.fromkeys(grp["cond"] + grp["sensor_raw"] + grp["sensor_causal"]))
log(f"프레임 {len(lab0)}, 실험 {len(set(g))}, S2 피처 {len(S2)}")
folds = list(GroupKFold(5, shuffle=True, random_state=0).split(X, y, g))


def f1(yt, yp):
    return C.macro_f1(yt, yp)


def lgb_fit_pred(Xtr, ytr, Xte, n_est=150, lr=0.1):
    p = P_.lgb_params(C.SEED, lr); p["n_jobs"] = 8
    m = lgb.train(p, lgb.Dataset(Xtr, label=ytr - 1, weight=P_.class_weights(ytr, "balanced")), num_boost_round=n_est)
    return m.predict(Xte)


# =================================================================== A. 스케일링 / 상대화
log("=== A. 피처 스케일링·상대화 ===")
ABS = {"meas_v", "meas_t1", "meas_t2", "meas_p", "atmp_t", "thermal_t", "meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o",
       "meas_no", "meas_no2", "meas_so2", "indu_t", "t1_minus_t2", "t1_minus_amb", "thermal_minus_amb", "thermal_minus_t1", "gas_sum_log"}
def is_abs(c):  # 절대 수준 피처(원 값, rolling mean/max, cummax)
    base = c.split("__")[0]; kind = c.split("__")[1] if "__" in c else ""
    return base in ABS and (kind == "" or kind.startswith("mean") or kind.startswith("max") or kind == "cummax")
REL = [c for c in S2 if (c in C.COND) or not is_abs(c)]
REL = [c for c in REL if c != "relative_time_s"]
log(f"절대 수준 피처 {sum(is_abs(c) for c in S2)}개 제외 -> 상대·변화량 피처 {len(REL)}개")


class Winsor:
    """학습 fold 에서 분위수 계산(0.5/99.5%) 후 클리핑 + 비음수 heavy-tail 컬럼 log1p."""
    def fit(self, Xa, y=None):
        Xa = np.asarray(Xa, float)
        self.lo = np.nanpercentile(Xa, 0.5, axis=0); self.hi = np.nanpercentile(Xa, 99.5, axis=0)
        mn = np.nanmin(Xa, axis=0); sk = pd.DataFrame(Xa).skew().to_numpy()
        self.logc = (mn >= 0) & (sk > 3)
        return self
    def transform(self, Xa):
        Xa = np.clip(np.asarray(Xa, float), self.lo, self.hi); Xa[:, self.logc] = np.log1p(Xa[:, self.logc]); return Xa
    def get_params(self, deep=True): return {}
    def set_params(self, **k): return self
    def fit_transform(self, Xa, y=None): return self.fit(Xa).transform(Xa)


def lr_run(cols, scaler, winsor, step=3):
    pred = np.zeros(len(y), int)
    for tr, va in folds:
        tr = tr[::step]
        steps = [("imp", SimpleImputer(strategy="median"))]
        if winsor: steps.append(("win", Winsor()))
        if scaler: steps.append(("sc", StandardScaler()))
        steps.append(("lr", LogisticRegression(C=1.0, max_iter=300, class_weight="balanced", n_jobs=8)))
        m = Pipeline(steps).fit(X.iloc[tr][cols], y[tr]); pred[va] = m.predict(X.iloc[va][cols])
    pc = C.causal_cummax_pred(lab0, pred)
    return f1(y, pred), f1(y, pc)

rows = []
t0 = time.time()
for name, cols, sc, win in [("LR | no scaling (raw)", S2, False, False), ("LR | StandardScaler", S2, True, False),
                            ("LR | winsorize + log1p + Standard", S2, True, True),
                            ("LR | winsor+log+Std, RELATIVE features only", REL, True, True)]:
    r_raw, r_cm = lr_run(cols, sc, win); rows.append(dict(model=name, n_feat=len(cols), f1_raw=r_raw, f1_cummax=r_cm)); log(f"{name:48s} raw {r_raw:.4f} cummax {r_cm:.4f}  ({time.time()-t0:.0f}s)")
# LightGBM: 스케일 불변성 + 절대/상대
def lgb_oof(cols, transform=None):
    oof = np.zeros((len(y), 6))
    for tr, va in folds:
        Xt, Xv = X.iloc[tr][cols], X.iloc[va][cols]
        if transform is not None:
            sc = StandardScaler().fit(Xt.fillna(Xt.median())); Xt = pd.DataFrame(sc.transform(Xt.fillna(Xt.median())), columns=cols); Xv = pd.DataFrame(sc.transform(Xv.fillna(Xt.median() * 0)), columns=cols)
            Xv = pd.DataFrame(sc.transform(X.iloc[va][cols].fillna(X.iloc[tr][cols].median())), columns=cols)
        oof[va] = lgb_fit_pred(Xt, y[tr], Xv)
    p = oof.argmax(1) + 1; return f1(y, p), f1(y, C.causal_cummax_pred(lab0, p))
for name, cols, tf in [("LightGBM | raw S2 (absolute+relative)", S2, None), ("LightGBM | StandardScaler S2 (scale invariance check)", S2, True),
                       ("LightGBM | RELATIVE features only", REL, None)]:
    r_raw, r_cm = lgb_oof(cols, tf); rows.append(dict(model=name, n_feat=len(cols), f1_raw=r_raw, f1_cummax=r_cm)); log(f"{name:48s} raw {r_raw:.4f} cummax {r_cm:.4f}  ({time.time()-t0:.0f}s)")
scal = pd.DataFrame(rows); scal.to_csv(OUT / "scaling.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4.2))
cols_ = ["#bbb" if m.startswith("LR | no") else ("#F58518" if m.startswith("LR") else "#4C78A8") for m in scal.model]
ax.barh(scal.model[::-1], scal.f1_cummax[::-1], color=cols_[::-1]); ax.set_xlim(0, 1); ax.set_xlabel("OOF macro-F1 (cummax, 5-fold by experiment)")
for i, v in enumerate(scal.f1_cummax[::-1]): ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_title("Feature scaling & relativization: critical for linear models, invisible to trees"); plt.tight_layout(); plt.savefig(OUT / "fig_scaling.png", dpi=130); plt.close()

# =================================================================== B. 이미지 가공 방식
log("=== B. 이미지 가공 방식 (이미지만으로의 판별력) ===")
STEP = 3 if not a.smoke else 5
lab0["fidx"] = lab0.groupby("experiment_id").cumcount()
sub = lab0[lab0.fidx % STEP == 0].copy()  # 각 실험의 첫 프레임 포함(fidx=0)
log(f"이미지 부표본 {len(sub)}장 (매 {STEP}번째 프레임)")


def variants(path):
    with Image.open(path) as im:
        rgb = im.convert("RGB"); a_ = np.asarray(rgb, dtype=np.float32) / 255.0
        small = np.asarray(rgb.resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
    gray = 0.299 * a_[..., 0] + 0.587 * a_[..., 1] + 0.114 * a_[..., 2]
    idx, white, valid = C._temp_index(a_)
    body = (idx > 0.12) & valid
    v = idx[body] if body.any() else idx.ravel()
    i2, w2, v2 = C._temp_index(small); vv = i2[v2] if v2.any() else i2.ravel()
    return dict(g_mean=float(gray.mean()), g_std=float(gray.std()), g_max=float(gray.max()), g_p95=float(np.percentile(gray, 95)), g_p99=float(np.percentile(gray, 99)), g_hot=float((gray >= .8).mean()),
                m_body_frac=float(body.mean()), m_p50=float(np.percentile(v, 50)), m_p90=float(np.percentile(v, 90)), m_p99=float(np.percentile(v, 99)), m_mean=float(v.mean()),
                r_mean=float(vv.mean()), r_p90=float(np.percentile(vv, 90)), r_p99=float(np.percentile(vv, 99)))
t0 = time.time()
vr = Parallel(n_jobs=12, batch_size=32)(delayed(variants)(p) for p in sub.img_file)
V = pd.DataFrame(vr, index=sub.index)
log(f"이미지 변형 피처 계산 {time.time()-t0:.0f}s")
sub = pd.concat([sub, V], axis=1)
ci = img_cache.reindex(sub.sample_id.values); ci.index = sub.index
idx_cols = [c for c in ci.columns if c.startswith("img_")]
sub = pd.concat([sub, ci[idx_cols + C.SIZE_FEATS]], axis=1)
m_cols = [c for c in sub.columns if c.startswith("m_")]
rel_base = idx_cols + m_cols
rel = sub[rel_base] - sub.groupby("experiment_id")[rel_base].transform("first")
rel.columns = [c + "__rel1st" for c in rel_base]
sub = pd.concat([sub, rel], axis=1)
SETS = {
    "gray stats (baseline v7)": [c for c in sub.columns if c.startswith("g_")],
    "RGB channel means": ["img_r", "img_g", "img_b"],
    "temp-index stats (abs)": idx_cols,
    "64x64 resize, temp-index (abs)": [c for c in sub.columns if c.startswith("r_")],
    "body-mask stats (abs)": m_cols,
    "temp-index + body-mask (abs)": idx_cols + m_cols,
    "relative to 1st frame (idx+mask)": list(rel.columns),
    "abs + relative": idx_cols + m_cols + list(rel.columns),
    "image SIZE only (shortcut)": C.SIZE_FEATS,
}
ys, gs = sub.label.to_numpy(), sub.experiment_id.to_numpy()
fo = list(GroupKFold(5, shuffle=True, random_state=0).split(sub, ys, gs))
def img_cv(cols):
    oof = np.zeros((len(ys), 6))
    for tr, va in fo:
        oof[va] = lgb_fit_pred(sub.iloc[tr][cols], ys[tr], sub.iloc[va][cols], n_est=120)
    p = oof.argmax(1) + 1
    return f1(ys, p), P_.per_class_f1(ys, p), p
irows, preds = [], {}
for name, cols in SETS.items():
    s, pcf, p = img_cv(cols); preds[name] = p
    irows.append(dict(image_features=name, n_feat=len(cols), macro_f1=s, **{f"L{i+1}": v for i, v in enumerate(pcf)})); log(f"{name:36s} n={len(cols):3d} F1={s:.4f}")
imgdf = pd.DataFrame(irows); imgdf.to_csv(OUT / "image_variants.csv", index=False)

# 센서에 이미지 가공별로 더했을 때의 증분 (실험별 정확도 차이 부트스트랩)
Xs = X.loc[sub.index, S2]
def sens_plus(extra_cols):
    cols_all = list(Xs.columns) + list(extra_cols)
    Z = pd.concat([Xs, sub[list(extra_cols)]], axis=1) if extra_cols else Xs
    oof = np.zeros((len(ys), 6))
    for tr, va in fo:
        oof[va] = lgb_fit_pred(Z.iloc[tr], ys[tr], Z.iloc[va], n_est=120)
    return oof.argmax(1) + 1
base_p = sens_plus([])
srows = [dict(added="(sensor only)", macro_f1=f1(ys, base_p), delta=0.0, ci_lo=0.0, ci_hi=0.0)]
rng = np.random.default_rng(0)
acc_base = pd.Series(base_p == ys).groupby(gs).mean()
for name in ["gray stats (baseline v7)", "temp-index stats (abs)", "temp-index + body-mask (abs)", "relative to 1st frame (idx+mask)"]:
    p = sens_plus(SETS[name]); acc = pd.Series(p == ys).groupby(gs).mean(); d_ = (acc - acc_base).to_numpy()
    bs = [d_[rng.integers(0, len(d_), len(d_))].mean() for _ in range(2000)]
    srows.append(dict(added="+ " + name, macro_f1=f1(ys, p), delta=f1(ys, p) - f1(ys, base_p), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5))))
    log(f"sensor {srows[-1]['added']:44s} F1={srows[-1]['macro_f1']:.4f} (Δ{srows[-1]['delta']:+.4f}) exp-acc Δ CI[{srows[-1]['ci_lo']:+.4f},{srows[-1]['ci_hi']:+.4f}]")
spd = pd.DataFrame(srows); spd.to_csv(OUT / "sensor_plus_image.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
d1 = imgdf.sort_values("macro_f1")
ax[0].barh(d1.image_features, d1.macro_f1, color=["#bbb" if "SIZE" in n else "#4C78A8" for n in d1.image_features]); ax[0].set_xlim(0, 1); ax[0].set_xlabel("image-only macro-F1 (5-fold by experiment)"); ax[0].set_title("How the image is processed changes what it can tell")
for i, v in enumerate(d1.macro_f1): ax[0].text(v + .005, i, f"{v:.3f}", va="center", fontsize=8)
d2 = spd.iloc[1:]; ax[1].errorbar(d2.delta, range(len(d2)), xerr=[d2.delta - d2.delta, d2.delta - d2.delta], fmt="o"); ax[1].barh(range(len(d2)), d2.delta, color=["#54A24B" if lo > 0 else "#bbb" for lo in d2.ci_lo])
ax[1].set_yticks(range(len(d2))); ax[1].set_yticklabels(d2.added, fontsize=8); ax[1].axvline(0, c="k", lw=.6); ax[1].set_xlabel("Δ macro-F1 over sensor-only"); ax[1].set_title("Added on top of sensors (green = per-experiment CI > 0)")
plt.tight_layout(); plt.savefig(OUT / "fig_image_variants.png", dpi=130); plt.close()
log("완료")
(OUT / "done.json").write_text(json.dumps({"ok": True}))
