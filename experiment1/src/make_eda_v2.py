"""eda_v2.ipynb 생성: 실험 단위·사건 기준 시간축으로 EDA 를 처음부터 다시.
1) 실험 지도(사건 정렬)  2) 경계 사전  3) 경계 주변 모달리티 정보 검정  4) 경계 변위·천장 추정
실행: python src/make_eda_v2.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# EDA v2 — 실험 단위 · 사건 기준 시간축으로 다시 보기

**질문**: (1) 라벨 경계는 어떤 신호에 묶여 있고 경계마다 성격이 어떻게 다른가, (2) 이미지는 센서가 모르는 정보를 경계 근처에서 주는가,
(3) 현재 오류는 '경계 근처의 작은 어긋남'인가 '구간 전체의 체계적 오분류'인가, 그 천장은 어디인가.

**원칙**: 분석 단위 = 실험(97개). 시계 시간 대신 **T1≥45℃ 도달 시각을 0으로 맞춘 사건 시간축**을 사용(분석용; 모델 입력이 아님).
경계 판별력은 경계 전후 ±24프레임(2분) 창에서만 평가한다(오류가 거기서 나기 때문).
""")

code(r"""
import sys, os, json, warnings
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score
import common as C, events as E, rules as R
sns.set_theme(style="whitegrid", context="notebook"); plt.rcParams.update({"figure.dpi": 105, "axes.titlesize": 10})
LCL = sns.color_palette("viridis_r", 6); LC = dict(zip(range(1, 7), LCL))
BCN = {1: "cyl", 2: "pouch", 3: "prism"}
N = 24   # 경계 주변 창 (프레임, 5초 간격 -> 2분)

lab0 = pd.concat([C.load_split(s) for s in ("train", "validation")]).sort_values(["experiment_id", "relative_time_s"], kind="mergesort").reset_index(drop=True)
img_cache = pd.concat([C.load_image_cache(s) for s in ("train", "validation", "test")])
lab = lab0.merge(img_cache.reset_index().rename(columns={"index": "sample_id"}), on="sample_id", how="left")
assert (lab.sample_id.values == lab0.sample_id.values).all()
ev = E.event_features(lab0); e1 = R.f_e1(lab0)
lab = pd.concat([lab, ev, e1], axis=1)
lab["tcross"] = lab.groupby("experiment_id").e1_tcross45.transform("max")
lab["ta"] = lab.relative_time_s - lab.tcross                       # 사건 정렬 시간 (T1=45℃ 도달 = 0)
lab["thermal_rise"] = lab.thermal_t - lab.groupby("experiment_id").thermal_t.transform("first")
lab["gas_sum"] = np.log1p(lab[C.GAS].sum(axis=1))
lab["t1_minus_t2"] = lab.meas_t1 - lab.meas_t2
lab["thermal_minus_t1"] = lab.thermal_t - lab.meas_t1
def slope(col, w=30):
    out = np.full(len(lab), np.nan); t = lab.relative_time_s.to_numpy(float); x = lab[col].to_numpy(float)
    for _, ix in lab.groupby("experiment_id", sort=False).indices.items():
        tt = t[ix]; lag = np.searchsorted(tt, tt - w, side="right") - 1; ok = lag >= 0
        o = np.full(len(ix), np.nan); o[ok] = (x[ix][ok] - x[ix][lag[ok]]) / np.maximum(tt[ok] - tt[lag[ok]], 1); out[ix] = o
    return out
lab["t1_slope30"] = slope("meas_t1"); lab["thermal_slope30"] = slope("thermal_t"); lab["p_slope30"] = slope("meas_p")
lab["grp"] = lab.bat_class.map(BCN) + "|tr" + lab.tr_method.astype(str)
exp = lab.groupby("experiment_id").agg(grp=("grp", "first"), bat_class=("bat_class", "first"), tr=("tr_method", "first"), cc=("bat_cc", "first"), n=("label", "size"), tcross=("tcross", "first"))
print("실험", len(exp), "| 프레임", len(lab), "| T1=45℃ 미도달 실험:", int(exp.tcross.isna().sum()))
display(exp.groupby("grp").agg(experiments=("n", "size"), frames=("n", "sum"), median_len=("n", "median"), median_tcross_s=("tcross", "median")))
""")

# ----------------------------------------------------------------- 1. 지도
md("## 1. 실험 지도 — 사건 정렬 시간축에서 그룹별 궤적")
code(r"""
sigs = [("ev_t1_rise", "T1 rise (C)"), ("thermal_rise", "thermal_t rise (C)"), ("ev_v_ratio", "V / V0"), ("ev_p_rise", "P rise"), ("gas_sum", "log1p(sum gas)"), ("img_p99", "image p99 temp-index")]
grps = [g for g in sorted(lab.grp.unique())]
fig, axs = plt.subplots(len(sigs), len(grps), figsize=(3.3 * len(grps), 2.2 * len(sigs)), sharex=True)
cols = np.array([LC[v] for v in lab.label])
for j, gname in enumerate(grps):
    m = (lab.grp == gname).to_numpy() & lab.ta.between(-300, 3000).to_numpy()
    for i, (c, ttl) in enumerate(sigs):
        a = axs[i, j]; y = lab[c].to_numpy(float)
        if c == "ev_v_ratio": y = np.clip(y, 0, 2)
        a.scatter(lab.ta[m], y[m], c=cols[m], s=1.2, alpha=.5, rasterized=True)
        if j == 0: a.set_ylabel(ttl, fontsize=8)
        if i == 0: a.set_title(f"{gname} (n_exp={int((exp.grp == gname).sum())})")
for a in axs[-1]: a.set_xlabel("time since T1>=45C (s)")
plt.suptitle("All experiments overlaid, event-aligned; color = label (light=L1 ... dark=L6)", y=1.0); plt.tight_layout(); plt.show()
""")
code(r"""
# 경계 시작 시각 (사건 정렬): 경계마다 그룹 간·실험 간 일관성
on = lab.groupby(["experiment_id", "label"]).head(1)
fig, ax = plt.subplots(1, 2, figsize=(17, 4.6))
sns.stripplot(data=on[on.label >= 2], x="label", y="ta", hue="grp", dodge=True, size=3, alpha=.7, ax=ax[0]); ax[0].set_yscale("symlog", linthresh=60); ax[0].set_title("Onset time of each label (since T1>=45C, symlog)"); ax[0].legend(fontsize=6, ncol=2)
on2 = on[on.label >= 2].copy(); on2["T1_at_onset"] = on2.meas_t1
sns.stripplot(data=on2, x="label", y="T1_at_onset", hue="tr_method", dodge=True, size=3, alpha=.7, ax=ax[1]); ax[1].set_yscale("symlog", linthresh=100); ax[1].set_title("T1 at label onset (by tr_method)")
plt.tight_layout(); plt.show()
tab = on[on.label >= 2].groupby(["label", "grp"]).ta.agg(n="size", median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75)).round(0)
display(tab.unstack(1)["median"].T)
""")

# ----------------------------------------------------------------- 2. 경계 사전
md(r"""
## 2. 경계 사전 — 어떤 신호가 어떤 경계를 알려 주는가
각 경계 k(= L(k-1)→Lk, 연속 전이만)에서 경계 전 N프레임 vs 후 N프레임을 비교한다.
- **pooled AUC**: 모든 실험을 섞어 계산 (신호 값이 실험 간에 비교 가능한가)
- **within-experiment AUC**: 실험마다 계산해 평균 (같은 실험 안에서 방향이 일관된가)

within 은 높은데 pooled 가 낮으면 *신호는 있으나 실험 간 보정이 안 되는* 경우 → 상대 변화량(첫 관측 대비/직전 대비) 피처가 필요하다는 뜻.
""")
code(r"""
SENS = ["meas_t1", "meas_t2", "thermal_t", "meas_v", "ev_v_ratio", "meas_p", "ev_p_rise", "gas_sum", "meas_co2", "meas_co", "meas_hf", "meas_hcn", "t1_minus_t2", "thermal_minus_t1",
        "ev_t1_rise", "thermal_rise", "t1_slope30", "thermal_slope30", "p_slope30", "e1_since45", "ev_t1_since_newhigh"]
IMGS = ["img_mean", "img_std", "img_max", "img_p50", "img_p90", "img_p99", "img_white", "img_hot67", "img_hot83", "img_cold20", "img_edge", "img_center_mean", "img_center_minus_all"]
lab["fidx"] = lab.groupby("experiment_id").cumcount()
def window(k):
    rows = []
    for eid, g in lab.groupby("experiment_id", sort=False):
        l = g.label.to_numpy(); pos = np.where((l[1:] == k) & (l[:-1] == k - 1))[0]
        if len(pos) != 1: continue
        b = pos[0] + 1; idx = g.index.to_numpy()
        pre = idx[max(0, b - N):b]; post = idx[b:b + N]
        rows.append((eid, pre, post))
    return rows
W = {k: window(k) for k in range(2, 7)}
print({k: len(v) for k, v in W.items()}, "<- 경계별 사용 가능 실험 수 (연속 전이만)")
def aucs(sig):
    po, wi = {}, {}
    for k, rows in W.items():
        ys, xs, es, wl = [], [], [], []
        for eid, pre, post in rows:
            x = lab.loc[np.r_[pre, post], sig].to_numpy(float); y = np.r_[np.zeros(len(pre)), np.ones(len(post))]
            ok = ~np.isnan(x)
            if ok.sum() and len(set(y[ok])) == 2: wl.append(roc_auc_score(y[ok], x[ok]))
            xs.append(x[ok]); ys.append(y[ok])
        x = np.concatenate(xs); y = np.concatenate(ys)
        po[k] = roc_auc_score(y, x) if len(set(y)) == 2 else np.nan; wi[k] = np.mean(wl) if wl else np.nan
    return po, wi
P_ = {s: aucs(s) for s in SENS + IMGS}
pooled = pd.DataFrame({s: P_[s][0] for s in P_}).T; within = pd.DataFrame({s: P_[s][1] for s in P_}).T
pooled.columns = within.columns = [f"L{k-1}->L{k}" for k in range(2, 7)]
fig, ax = plt.subplots(1, 2, figsize=(14, 11))
for a, d, t in [(ax[0], pooled, "pooled AUC (cross-experiment)"), (ax[1], within, "within-experiment AUC (mean)")]:
    sns.heatmap(d, annot=True, fmt=".2f", cmap="RdBu_r", center=.5, vmin=0, vmax=1, ax=a, cbar=False, annot_kws={"size": 7}); a.set_title(t); a.axhline(len(SENS), c="k", lw=2)
ax[1].set_yticklabels([]); plt.suptitle(f"Boundary detectability, +/-{N} frames. rows above bold line = sensor-derived, below = image-derived (0.5 = none)", y=1.0); plt.tight_layout(); plt.show()
gap = (within.sub(0.5).abs() - pooled.sub(0.5).abs()).round(2)
print("보정 격차 |within-0.5| - |pooled-0.5| 이 큰 신호 (신호는 있으나 실험 간 비교 불가):"); display(gap.stack().sort_values(ascending=False).head(12).rename("gap").to_frame())
""")
md(r"""
> **읽는 법**: AUC > 0.5 는 경계 이후 값이 커짐, < 0.5 는 작아짐. 센서/이미지 행 중 어느 경계에서 어느 쪽이 먼저 신호를 주는지 비교한다.
> L1→L2 는 T1 규칙(결정적), 이후 경계는 가장 강한 신호가 경계마다 다른지 확인한다.
""")

# ----------------------------------------------------------------- 3. 모달리티 정보 검정
md(r"""
## 3. 경계 주변에서 센서 · 이미지 · 둘 다의 판별력 (실험 단위 5-fold, 인과 피처)
각 경계의 ±N프레임 샘플로 '경계 후인가?' 를 맞춘다. 같은 실험은 같은 fold. 실험 단위 부트스트랩으로 AUC 신뢰구간과 (둘 다 − 센서) 차이를 본다.
""")
code(r"""
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
X = C.build_feature_table(lab0, img_cache)
grp_cols = C.feature_groups(list(X.columns))
FS = {"sensor": list(dict.fromkeys(grp_cols["cond"] + grp_cols["sensor_raw"] + grp_cols["sensor_causal"])),
      "image": list(dict.fromkeys(grp_cols["cond"] + grp_cols["img_raw"] + grp_cols["img_causal"]))}
FS["both"] = list(dict.fromkeys(FS["sensor"] + FS["image"]))
def boundary_cv(k):
    rows = W[k]; idx = np.concatenate([np.r_[p, q] for _, p, q in rows]); y = np.concatenate([np.r_[np.zeros(len(p)), np.ones(len(q))] for _, p, q in rows])
    g = np.concatenate([[e] * (len(p) + len(q)) for e, p, q in rows]); oof = {n: np.zeros(len(y)) for n in FS}
    for tr, va in GroupKFold(5).split(idx, y, g):
        for n, cols in FS.items():
            m = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.08, num_leaves=15, min_child_samples=15, colsample_bytree=.5, subsample=.8, subsample_freq=1, verbose=-1, n_jobs=4)
            m.fit(X.loc[idx[tr], cols], y[tr]); oof[n][va] = m.predict_proba(X.loc[idx[va], cols])[:, 1]
    return y, g, oof
def boot_auc(y, g, oof, B=300):
    ug = np.unique(g); gi = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(0); res = {n: [] for n in oof}; res["both-sensor"] = []
    for _ in range(B):
        ii = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))]); yy = y[ii]
        if len(set(yy)) < 2: continue
        a = {n: roc_auc_score(yy, oof[n][ii]) for n in oof}
        for n in oof: res[n].append(a[n])
        res["both-sensor"].append(a["both"] - a["sensor"])
    return {n: (np.mean(v), *np.percentile(v, [2.5, 97.5])) for n, v in res.items()}
rows = []
for k in range(2, 7):
    y, g, oof = boundary_cv(k); b = boot_auc(y, g, oof)
    for n, (m, lo, hi) in b.items(): rows.append(dict(boundary=f"L{k-1}->L{k}", model=n, auc=m, lo=lo, hi=hi, n_exp=len(np.unique(g))))
mod = pd.DataFrame(rows); mod.to_csv(C.OUT / "eda_v2_modality.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.4))
for n, c in zip(["sensor", "image", "both"], ["#4C78A8", "#F58518", "#54A24B"]):
    d = mod[mod.model == n]; ax[0].errorbar(np.arange(5) + {"sensor": -.12, "image": 0, "both": .12}[n], d.auc, yerr=[d.auc - d.lo, d.hi - d.auc], fmt="o", color=c, label=n, capsize=3)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(sorted(mod.boundary.unique())); ax[0].axhline(.5, c="k", lw=.6); ax[0].set_ylabel("OOF AUC (before vs after boundary)"); ax[0].set_title("Boundary discrimination by modality (95% CI over experiments)"); ax[0].legend()
d = mod[mod.model == "both-sensor"]; ax[1].bar(range(5), d.auc, yerr=[d.auc - d.lo, d.hi - d.auc], color=["#54A24B" if l > 0 else "#bbb" for l in d.lo], capsize=4); ax[1].axhline(0, c="k", lw=.6)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(sorted(mod.boundary.unique())); ax[1].set_title("AUC gain of image over sensor-only (green = CI excludes 0)")
plt.tight_layout(); plt.show(); display(mod.pivot(index="boundary", columns="model", values="auc").round(3))
""")

# ----------------------------------------------------------------- 4. 변위 / 천장
md(r"""
## 4. 오류의 구조와 천장 추정
현재 센서 모델의 OOF 예측(`outputs/oof_gate.parquet`, 실험 단위 5-fold, cummax 후처리)로
(a) 경계 변위(예측 시작 − 정답 시작, 프레임), (b) 오류 프레임이 정답 경계에서 얼마나 떨어져 있는지,
(c) 경계를 ±k프레임 어긋나게 했을 때의 Macro-F1(경계 지터 곡선)을 본다.
""")
code(r"""
oof = pd.read_parquet(C.OUT / "oof_gate.parquet")[["sample_id", "pred_sensor"]]
d = lab[["sample_id", "experiment_id", "relative_time_s", "label", "fidx"]].merge(oof, on="sample_id", how="left")
assert d.pred_sensor.notna().all()
d["pred_cm"] = C.causal_cummax_pred(d.rename(columns={}), d.pred_sensor.to_numpy())
print(f"OOF pooled macro-F1  raw={C.macro_f1(d.label, d.pred_sensor):.4f}  cummax={C.macro_f1(d.label, d.pred_cm):.4f}")
# (a) 경계 변위
rows = []
for eid, g in d.groupby("experiment_id", sort=False):
    l = g.label.to_numpy(); p = g.pred_cm.to_numpy()
    for k in range(2, 7):
        ti = np.where(l == k)[0]
        if len(ti) == 0 or (l == k - 1).sum() == 0: continue
        pi = np.where(p >= k)[0]
        rows.append(dict(experiment_id=eid, boundary=f"L{k-1}->L{k}", disp=(pi[0] - ti[0]) if len(pi) else np.nan, never=len(pi) == 0))
disp = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(16, 4.4))
sns.stripplot(data=disp.assign(disp=disp.disp.clip(-300, 300)), x="boundary", y="disp", size=3, alpha=.6, ax=ax[0]); sns.boxplot(data=disp.assign(disp=disp.disp.clip(-300, 300)), x="boundary", y="disp", color="#ddd", fliersize=0, ax=ax[0], zorder=0)
ax[0].axhline(0, c="k", lw=.7); ax[0].set_ylabel("predicted onset - true onset (frames, clipped +/-300)"); ax[0].set_title("Boundary displacement per experiment (+ = late)")
summ = disp.groupby("boundary").disp.agg(n="size", median="median", mad=lambda s: (s - s.median()).abs().median(), within6=lambda s: (s.abs() <= 6).mean(), within24=lambda s: (s.abs() <= 24).mean(), gt100=lambda s: (s.abs() > 100).mean())
display(summ.round(2))
# (b) 오류 프레임의 경계까지 거리
dist = np.full(len(d), np.nan); gi = d.groupby("experiment_id", sort=False).indices
for eid, ix in gi.items():
    l = d.label.to_numpy()[ix]; bpos = np.where(np.diff(l) != 0)[0] + 1
    dist[ix] = np.abs(np.arange(len(ix))[:, None] - bpos[None, :]).min(1) if len(bpos) else 9999
d["dist"] = dist; err = d[d.pred_cm != d.label]
bins = [-1, 6, 12, 24, 60, 120, 1e9]; lab_b = ["<=6", "7-12", "13-24", "25-60", "61-120", ">120"]
share = pd.cut(err.dist, bins, labels=lab_b).value_counts(normalize=True).reindex(lab_b)
ax[1].bar(lab_b, share.values, color="#4C78A8"); ax[1].set_title(f"Where errors sit: distance (frames) to nearest TRUE boundary  (n_err={len(err)})"); ax[1].set_ylabel("share of error frames")
for i, v in enumerate(share.values): ax[1].text(i, v + .005, f"{v:.0%}", ha="center")
plt.tight_layout(); plt.show()
print(f"오류 프레임 중 정답 경계 ±24프레임(2분) 이내: {(err.dist <= 24).mean():.1%} | 60프레임(5분) 초과: {(err.dist > 60).mean():.1%}")
""")
code(r"""
# (c) 경계 지터 곡선: 실제 라벨 시퀀스에서 각 경계를 U(-k,k) 프레임 어긋나게 한 예측의 Macro-F1
rng = np.random.default_rng(0)
seqs = {eid: g.label.to_numpy() for eid, g in d.groupby("experiment_id", sort=False)}
def jitter_f1(k, draws=15):
    out = []
    for _ in range(draws):
        ys, ps = [], []
        for l in seqs.values():
            ch = np.where(np.diff(l) != 0)[0] + 1; incs = l[ch] - l[ch - 1]
            sh = np.sort(np.clip(ch + rng.integers(-k, k + 1, len(ch)), 1, len(l) - 1)) if k > 0 else ch
            p = np.full(len(l), l[0]);
            for s, inc in zip(sh, incs): p[s:] += inc
            ys.append(l); ps.append(p)
        out.append(C.macro_f1(np.concatenate(ys), np.concatenate(ps)))
    return np.mean(out)
ks = [0, 1, 2, 3, 5, 8, 12, 18, 25, 40, 60, 100]
curve = pd.Series({k: jitter_f1(k) for k in ks})
f1_oof = C.macro_f1(d.label, d.pred_cm)
fig, ax = plt.subplots(figsize=(8.5, 4.6)); ax.plot(curve.index, curve.values, "o-"); ax.set_xscale("symlog", linthresh=3)
ax.axhline(f1_oof, c="C3", ls="--", label=f"our OOF (cummax) {f1_oof:.3f}"); ax.axhline(0.9447, c="C2", ls=":", label="reference top score 0.9447 (different split)")
ax.set_xlabel("max boundary shift k (frames, x5 s)"); ax.set_ylabel("macro-F1 if every boundary is off by U(-k,k)"); ax.set_title("What boundary accuracy does a given F1 imply?"); ax.legend(); plt.tight_layout(); plt.show()
def k_at(f):
    s = curve.sort_index(); return float(np.interp(f, s.values[::-1], s.index.values[::-1]))
print(curve.round(4).to_dict())
print(f"our OOF F1 {f1_oof:.3f} ~ 경계 지터 k≈{k_at(f1_oof):.0f}프레임(±{5*k_at(f1_oof):.0f}s) | 참고 0.9447 ~ k≈{k_at(0.9447):.0f}프레임(±{5*k_at(0.9447):.0f}s)")
""")
# ----------------------------------------------------------------- 5. 팀 분석 종합 vs 재계산
md(r"""
## 5. 팀 분석 종합(외부 입력) 과 이 노트북의 재계산 비교
아래 첫 표는 팀원이 전달한 97개 실험 OOF 집계(**외부 입력, 이 노트북 코드로 재현한 값이 아님**)이고, 두 번째부터는 이 저장소의 센서 모델 OOF(`outputs/oof_gate.parquet`, 전체 길이, cummax)로 같은 항목을 다시 계산한 값이다.
두 OOF 는 모델 설정·피처·fold 가 달라 **절대 수치는 직접 비교하지 않고, 패턴(어디가 약한가)이 같은지**만 본다.

**팀 집계 (센서 / 이미지 F1)**

| | L1 | L2 | L3 | L4 | L5 | L6 | 전체 |
|---|---|---|---|---|---|---|---|
| 센서 / 이미지 | 1.00 / 0.97 | 0.82 / 0.75 | 0.90 / 0.83 | 0.90 / 0.65 | 0.92 / 0.68 | 0.89 / 0.79 | 0.905 / 0.778 |

| bat_class | 센서 전체 | 단계별(L1~L6) | 이미지 전체 | 오류 비중 / 행 비중 |
|---|---|---|---|---|
| 원통형(1) | 0.909 | 1.0, 0.82, 0.91, 0.83, 0.96, 0.95 | 0.735 | 23% / 19% |
| 파우치(2) | 0.806 | 1.0, 0.74, 0.88, 0.54, 0.74, 0.93 | 0.721 | 34% / 40% |
| 각형(3) | 0.906 | 1.0, 0.86, 0.92, 0.94, 0.88, 0.83 | 0.763 | 43% / 41% |

팀 보고 요약: 실험별 센서 F1 평균 0.83 / 중앙값 0.88 / 최소 0.43, 97개 중 32개가 0.8 미만(파우치 21/33, 각형 9/33, 원통형 2/31). 오류 상위 10개 실험이 오류의 45%.
파우치 L4 는 33개 실험에 125행, L5 는 81행뿐. 조건 칸(class, cc, soc, tr) 하나에 실험이 1~8개. Test 11개 = 원통형 가열 5, 파우치 가열 1, 파우치 과충전 1, 각형 가열 2, 각형 과충전 2.
임베딩(DINOv2 PCA) 결과: PCA4 +0.0026(3/5 fold), PCA8 +0.0020, PCA16 −0.0052 → 잡음 범위. 파우치만 보면 PCA4/PCA8 에서 0.806→0.821(+0.015), 우연 여부 확인 필요.
""")
code(r"""
from sklearn.metrics import f1_score
dd = d.merge(exp[["bat_class", "tr", "grp"]].reset_index(), on="experiment_id")
dd["err"] = dd.pred_cm != dd.label
lbls = list(range(1, 7))
f_lab = f1_score(dd.label, dd.pred_cm, labels=lbls, average=None, zero_division=0)
print("이 노트북 OOF 단계별 F1:", dict(zip([f"L{i}" for i in lbls], f_lab.round(2))), "| 전체", round(C.macro_f1(dd.label, dd.pred_cm), 3))
rows = []
for bc in (1, 2, 3):
    s = dd[dd.bat_class == bc]; fl = f1_score(s.label, s.pred_cm, labels=lbls, average=None, zero_division=0)
    e4 = s[s.label == 4]; e5 = s[s.label == 5]
    rows.append(dict(bat_class=BCN[bc], experiments=s.experiment_id.nunique(), macro_F1=round(float(np.mean(fl)), 3), **{f"L{i}": round(v, 2) for i, v in zip(lbls, fl)},
                     err_share=round(s.err.sum() / dd.err.sum(), 2), row_share=round(len(s) / len(dd), 2),
                     L4_exps=e4.experiment_id.nunique(), L4_rows=len(e4), L5_exps=e5.experiment_id.nunique(), L5_rows=len(e5)))
tab_bc = pd.DataFrame(rows).set_index("bat_class"); display(tab_bc)
pe = dd.groupby("experiment_id").apply(lambda s: pd.Series({"f1": f1_score(s.label, s.pred_cm, labels=sorted(set(s.label) | set(s.pred_cm)), average="macro", zero_division=0),
                                                            "bat_class": s.bat_class.iloc[0], "err": int(s.err.sum()), "grp": s.grp.iloc[0]}))
print(f"실험별 F1: 평균 {pe.f1.mean():.2f} / 중앙값 {pe.f1.median():.2f} / 최소 {pe.f1.min():.2f} | 0.8 미만 {int((pe.f1 < .8).sum())}/{len(pe)}  (클래스별 " +
      ", ".join(f"{BCN[b]} {int((pe[pe.bat_class == b].f1 < .8).sum())}/{int((pe.bat_class == b).sum())}" for b in (1, 2, 3)) + ")")
top = pe.sort_values("err", ascending=False).head(10)
print(f"오류 상위 10개 실험의 오류 비중: {top.err.sum() / pe.err.sum():.0%} | 구성:", top.grp.value_counts().to_dict())
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
sns.barplot(x=[f"L{i}" for i in lbls], y=f_lab, color="#4C78A8", ax=ax[0]); ax[0].set_ylim(0, 1.05); ax[0].set_title("Per-label F1 (this repo, OOF)")
sns.stripplot(data=pe.reset_index(), x=pe.reset_index().bat_class.map(BCN), y="f1", size=4, alpha=.7, ax=ax[1]); ax[1].axhline(.8, c="r", ls="--", lw=.8); ax[1].set_title("Per-experiment macro-F1 by bat_class")
ex = pe.groupby("grp").err.sum().sort_values(); ax[2].barh(ex.index, ex.values / ex.sum(), color="#F58518"); ax[2].set_title("Error share by condition group")
plt.tight_layout(); plt.show()
""")
md(r"""
> **비교 결과의 읽는 법**: 두 OOF 에서 같은 패턴이 반복되면(L2 가 가장 약함, 파우치가 약함, 오류가 소수 실험에 집중) 그건 모델 설정과 무관한 **데이터의 구조적 특성**이다.
> 파우치 L4/L5 는 학습 가능한 실험·행이 근본적으로 적어 매크로 평균을 끌어내린다.
""")

# ----------------------------------------------------------------- 6. 열화상 L2/L3 (외부)
md(r"""
## 6. 참조: L2·L3 열화상 비교 (팀원 분석, 외부 입력)
- L2 마지막 프레임과 L3 첫 프레임은 거의 같아 보이고, 이미지 피처는 L3 진입 약 150초 뒤부터 크게 오른다 → 전환 *순간*을 짚는 데는 약하다.
- 절대값 기준 L3 vs L2 AUC 는 0.56~0.73 으로 낮지만 **실험 내 AUC 는 0.92**까지 오른다 → 실험 첫 프레임 대비 상대 변화가 유효 (카메라 구도·크기가 13종으로 달라 절대값은 비교 불가).
- 가열 영역 면적(body_frac)과 상위 10% 온도(heat_p90)가 가장 정보가 많고, 최고온·흰색 비율은 L2/L3 구분에 쓸모가 없다.
- 2↔3 경계 ±30초 AUC: `meas_t1` 0.95, `thermal_t` 0.90, 이미지 최고 0.74 → 이 경계에서도 센서가 이미지를 압도.
이 노트북 3절의 결과(센서+이미지 − 센서의 CI 가 0 포함)와 일치한다.
""")

# ----------------------------------------------------------------- 7. 증거 -> 반영
md(r"""
## 7. 증거 → 학습/코드 반영

| 증거 | 반영 방법 | 위치 |
|---|---|---|
| L1→L2 는 T1=45℃ 규칙(97/97) | 추론에서 L1 을 규칙으로 고정, L1 프레임(43%)을 빼고 L2~L6 모델 학습 | 라운드 2 **E8** |
| L3→L4, L4→L5, L5→L6 센서가 ±6프레임 이내 81~97%, 손실은 L2→L3 에 집중 | L2 vs ≥L3 전문가 + L3~L6 모델의 캐스케이드 | 라운드 2 **E9** |
| Test 2/11 실험이 500행 재표집, 긴 실험이 프레임의 56% | 평가를 500행 상한으로 맞춤(반영됨), 학습 가중 w=min(1,500/n)과 500행 재표집 증강 | 평가: 라운드 1 / 학습: **E10** |
| 오류 58% 가 경계 ±24프레임 안 | 경계 근처 프레임 가중 ×2 | **E11** |
| 이미지는 모든 경계에서 센서에 정보 없음, 실험 내 상대 변화만 유효 | 기본 입력에서 제외, 시험 시 첫 프레임 대비 변화(body_frac, heat_p90)를 L2 vs ≥L3 전문가에만 | **E12** (마지막) |
| 오류가 소수 실험에 집중, 조건 칸당 실험 1~8개 (팀 분석) | 실험당 총 가중 균형(1/√n), 강한 정규화, 조건 크기(bat_cc, soc) 제거 | 라운드 1 **E4, E5, E7** |
| 센서 이득의 대부분이 실험 간 보정 문제(within>pooled) | 첫 관측/T1=45℃ 기준 상대·사건 시계 피처 | 라운드 1 **E1, E2, E3** |
| 팀 제안: 임베딩의 파우치 이득이 우연인지 | 실험 단위 짝지은 비교(시드 3, 실험별 ΔF1 부호 검정) — 이 저장소 프로토콜 밖, 팀 쪽 진행 | 팀 |

**반영하지 않는 것**: 센서·이미지 확률 결합(근거 없음), L3-1/L3-2 라벨 분리(레짐을 미리 맞힐 수 없음: 팀 AUC ≤0.73, 대신 레짐 간접 피처 E2).
**알려진 한계**: 오류의 24% 는 경계에서 60프레임 넘게 떨어진 구간 단위 오분류(예: 느린 가열 실험의 L5) → 피처·가중으로 풀리기 어려운 실험 단위 잠재 상태 문제.
""")

# ----------------------------------------------------------------- 8. 프로토콜
md(r"""
## 8. 규칙 적용 프로토콜 — 평가·채택 기준은 결과를 보기 전에 고정 (`PROTOCOL.md`)
- **평가**: Train+Val 97개 실험, 실험 단위 5-fold × 3반복, **Test 모사(실험당 500행 상한 + 피처 재계산)** 행에서 채점. 모델 하이퍼파라미터는 후보마다 튜닝하지 않음.
- **채택 기준(전부 충족)**: 주 지표 Δ ≥ +0.003 · 실험별 정확도 차이의 부트스트랩 95% CI 하한 > 0 · 3반복 중 2회 이상 개선 · 공식 Validation 부분이 −0.01 보다 나빠지지 않음 · 자유 파라미터 없음 · 인과성 검사 통과.
- **라운드 1**: E1 L2 시계 · E2 레짐 간접 피처 · E3 방식별 트리거 · E4 실험 균형 가중 · E5 조건 크기 제거 · E6 순서형+단조 제약 · E7 강한 정규화
- **라운드 2**(라운드 1 결과를 읽기 전에 등록): E8 L1 규칙+조건부 모델 · E9 L2→L3 캐스케이드 · E10 Test 모사 학습 · E11 경계 집중 가중 · E12 이미지 상대 변화(L2→L3 전문가)
- 변경 이력: 최초에는 전체 길이로 풀링 채점했으나 Test 구조(2/11 실험 500행 재표집) 때문에 Test 모사 평가로 교체하고 처음부터 재실행(후보·기준은 불변).
""")
code(r"""
pr = C.OUT / "protocol" / "results.csv"
if pr.exists():
    r = pd.read_csv(pr); print("라운드 1 결과 (Test 모사 평가, 채택은 코드가 자동 판정):"); display(r.round(4))
    dj = C.OUT / "protocol" / "decisions.json"
    if dj.exists():
        dec = json.loads(dj.read_text()); display(pd.DataFrame({k: {**{kk: v[kk] for kk in ("delta_primary", "delta_exp_acc", "ci95_lo", "ci95_hi", "wins", "delta_val", "improved_exps", "worsened_exps", "adopt")}} for k, v in dec.items()}).T)
else:
    print("프로토콜 라운드 1 결과가 아직 없습니다 (outputs/protocol/results.csv). run_protocol.sh 완료 후 이 셀을 다시 실행하세요.")
""")

md(r"""
## 결론 (실행 결과 기반)
CONCLUSIONS_PLACEHOLDER
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
p = ROOT / "src" / "eda_v2_conclusions.md"
text = p.read_text(encoding="utf-8") if p.exists() else "(실행 후 결론을 추가하세요)"
for c in nb.cells:
    if "CONCLUSIONS_PLACEHOLDER" in c.source:
        c.source = c.source.replace("CONCLUSIONS_PLACEHOLDER", text)
nbf.write(nb, ROOT / "eda_v2.ipynb")
print("생성:", ROOT / "eda_v2.ipynb")
