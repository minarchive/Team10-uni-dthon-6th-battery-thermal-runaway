"""experiment1.ipynb 생성 (EDA 시각화 + 모델 결과). 실행: python src/make_notebook.py  (cwd 무관)"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# =============================================================================
md(r"""
# Uni-DTHON 배터리 열폭주 6단계 — EDA 재해석 + LightGBM 모델 (experiment1)

**목표**: 이미 알려진 데이터 특성(Train/Test 분포 차이, bat_class별 이미지 크기 지름길, 클래스 불균형, 5/10/15초 간격)을 정량 시각화하고,
그 근거를 *피처/학습 설계*에 연결한 뒤 Validation Macro-F1 기준으로 모델을 선택하고 `submission.csv` 를 만든다.

**규칙 준수(미래 정보 금지)**: 모든 시계열 피처는 같은 `experiment_id` 의 현재·과거 관측만 사용 (cummax, 첫 관측 대비 변화, 시간 기준 rolling(30/60s), 시간차 delta(15/30/60s)).
실험 전체 통계·총 길이·종료 시각은 사용하지 않음. 후처리도 과거 예측만 사용(누적 최댓값/과거 k개 중앙값).

**구성**: 1) 데이터 개요 · 2) 라벨/분포 이동 · 3) 라벨 시간 구조 · 4) 센서 EDA · 5) 이미지 EDA · 6) 모델(ablation→선택→제출) · 7) 해석
""")

code(r"""
import sys, os, json, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import common as C

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 105, "axes.titlesize": 11, "axes.labelsize": 10})
LC = dict(zip(range(1, 7), sns.color_palette("viridis_r", 6)))        # 라벨 색 (단계가 높을수록 진하게)
LCL = [LC[i] for i in range(1, 7)]
SPC = {"train": "#4C78A8", "validation": "#F58518", "test": "#54A24B"}  # split 색
BCN = {1: "cylindrical", 2: "pouch", 3: "prismatic"}
BCC = {1: "#9467BD", 2: "#2CA02C", 3: "#D62728"}

raw = C.load_all()
tr, va, te = raw["train"], raw["validation"], raw["test"]
al = pd.concat([tr, va, te], ignore_index=True)
lab = pd.concat([tr, va], ignore_index=True)
print("DATA_DIR:", C.DATA_DIR)
pd.DataFrame({"rows": [len(tr), len(va), len(te)], "experiments": [d.experiment_id.nunique() for d in (tr, va, te)]}, index=["train", "validation", "test"])
""")

# ------------------------------------------------------------------ 1. 개요
md("## 1. 데이터 개요 — 실험(97+11개)이 실질 표본 단위")
code(r"""
exp = al.groupby("experiment_id").agg(split=("split", "first"), bat_class=("bat_class", "first"), bat_cc=("bat_cc", "first"),
        soc=("soc_status", "first"), tr=("tr_method", "first"), n=("sample_id", "size"), tmax=("relative_time_s", "max"))
exp["cond"] = exp.bat_class.astype(str) + "|" + exp.bat_cc.astype(str) + "|soc" + exp.soc.astype(str) + "|tr" + exp.tr.astype(str)

fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
ct = exp.groupby(["split", "bat_class"]).size().unstack(0).reindex(columns=["train", "validation", "test"]).fillna(0)
(ct / ct.sum()).plot.bar(ax=ax[0], color=[SPC[c] for c in ct.columns]); ax[0].set_title("Experiment share by bat_class (per split)")
ax[0].set_xticklabels([BCN[i] for i in ct.index], rotation=0); ax[0].set_ylabel("share of experiments")
for i, c in enumerate(ct.columns):
    for j, v in enumerate(ct[c]): ax[0].text(j + (i - 1) * 0.27, (v / ct[c].sum()) + 0.01, int(v), ha="center", fontsize=8)
cc = exp.groupby(["bat_cc", "split"]).size().unstack(1).reindex(columns=["train", "validation", "test"]).fillna(0)
cc.plot.bar(ax=ax[1], color=[SPC[c] for c in cc.columns], stacked=True); ax[1].set_title("Experiments per rated capacity (Ah)"); ax[1].set_ylabel("# experiments")
plt.tight_layout(); plt.show()

piv = exp.groupby(["cond", "split"]).agg(exps=("n", "size"), rows=("n", "sum")).unstack(1).fillna(0).astype(int)
display(piv)
""")
md(r"""
> **관찰**: Test는 11개 실험뿐이고 조건 조합이 Train과 다르게 분포한다. Validation(19개)도 실험 단위로는 작아서 점수 변동이 크다 →
> 모델 선택 시 Validation 단일 점수뿐 아니라 Train+Val 97개 실험 GroupKFold 점수를 병행 확인한다(6장).
""")

code(r"""
# 실험 길이 / 샘플링 간격 (Test 는 5/10/15초 혼재)
al = al.sort_values(["experiment_id", "relative_time_s"])
al["dt"] = al.groupby("experiment_id").relative_time_s.diff()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for s in ["train", "validation", "test"]:
    sns.histplot(exp[exp.split == s].n, bins=20, ax=ax[0], color=SPC[s], label=s, alpha=0.55, stat="count")
ax[0].set_title("Rows per experiment"); ax[0].legend()
g = al.dropna(subset=["dt"]).groupby(["split", "dt"]).size().unstack(0).fillna(0)
(g / g.sum()).plot.bar(ax=ax[1], color=[SPC[c] for c in g.columns]); ax[1].set_title("Sampling interval (dt) share"); ax[1].set_xlabel("dt (s)")
sns.boxplot(data=exp, x="split", y="tmax", order=["train", "validation", "test"], palette=SPC, ax=ax[2]); ax[2].set_title("Experiment duration (s)")
plt.tight_layout(); plt.show()
print("test dt share:", (te.sort_values(["experiment_id", "relative_time_s"]).groupby("experiment_id").relative_time_s.diff().value_counts(normalize=True)).round(3).to_dict())
print("test: 실험마다 dt 가 5/10/15 가 섞이는지 ->", te.sort_values(["experiment_id", "relative_time_s"]).groupby("experiment_id").relative_time_s.diff().groupby(te.experiment_id).nunique().to_dict())
""")
md(r"""
> **설계 근거**: Test 는 11개 실험 중 9개가 5초 간격이고, 2개 실험만 10/15초 간격이 섞여 있다 → 행 수(`rolling(k rows)`, `shift(k)`) 기반 피처는 train/test 에서 의미가 달라진다.
> 그래서 모든 시계열 피처를 **실제 경과시간(초)** 기준으로 계산하고, 학습 실험에서 프레임을 랜덤 skip(1~3칸)한 **간격 증강 사본**으로
> 같은 조건을 흉내 낸 뒤(건너뛴 행만으로 피처 재계산 → 미래 누수 없음) 6장에서 효과를 검증한다.
""")

# ------------------------------------------------------------------ 2. 라벨
md("## 2. 라벨 분포와 분포 이동")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
d = lab.groupby(["split", "label"]).size().unstack(0)
(d / d.sum()).plot.bar(ax=ax[0], color=[SPC["train"], SPC["validation"]]); ax[0].set_title("Label share: train vs validation"); ax[0].set_xlabel("label")
for i, c in enumerate(d.columns):
    for j, v in enumerate(d[c]): ax[0].text(j + (i - 0.5) * 0.3, v / d[c].sum() + 0.005, f"{v:,}", ha="center", fontsize=7, rotation=90)
x = lab.groupby(["bat_class", "label"]).size().unstack(1)
(x.div(x.sum(1), axis=0)).plot.bar(stacked=True, ax=ax[1], color=[LC[i] for i in range(1, 7)]); ax[1].set_title("Label mix within each bat_class (train+val)")
ax[1].set_xticklabels([BCN[i] for i in x.index], rotation=0); ax[1].legend(title="label", bbox_to_anchor=(1, 1))
y = lab.groupby(["tr_method", "label"]).size().unstack(1)
(y.div(y.sum(1), axis=0)).plot.bar(stacked=True, ax=ax[2], color=[LC[i] for i in range(1, 7)], legend=False); ax[2].set_title("Label mix by tr_method (1=heating, 2=overcharge)")
ax[2].set_xticklabels(["heating", "overcharge"], rotation=0)
plt.tight_layout(); plt.show()

# 라벨 x bat_class 행 수 (실험 수)
t1 = lab.groupby(["bat_class", "label"]).size().unstack(1).fillna(0).astype(int)
t2 = lab.groupby(["bat_class", "label"]).experiment_id.nunique().unstack(1).fillna(0).astype(int)
print("rows per (bat_class, label)"); display(t1.style.background_gradient(axis=None, cmap="Blues"))
print("experiments that contain the label"); display(t2.style.background_gradient(axis=None, cmap="Greens"))
""")
md(r"""
> **관찰**: bat_class 별 라벨 구성이 크게 다르다(예: bat_class 2 는 L1 비중이 매우 큼, L4~L6 이 드묾). 즉 *bat_class(=이미지 크기)가 라벨 prior* 를 바꾸므로
> 이미지 크기는 지름길이 된다. Test 의 bat_class 구성이 Train 과 다르므로, 이 prior 에 기대면 Test 에서 흔들린다 → B0(크기만)/B0c(조건만) 기준선 대비 이득을 반드시 확인.
""")

# ------------------------------------------------------------------ 3. 시간 구조
md("## 3. 라벨의 시간 구조 — 단조 비감소, 그러나 시작 시각 편차가 큼")
code(r"""
lab = lab.sort_values(["experiment_id", "relative_time_s"])
mono = lab.groupby("experiment_id").label.apply(lambda s: bool((s.diff().dropna() >= 0).all()))
jump = lab.groupby("experiment_id").label.apply(lambda s: int(s.diff().dropna().max()))
print(f"단조 비감소 실험: {mono.sum()}/{len(mono)} | 한 번에 건너뛰는 최대 단계 점프 분포:", jump.value_counts().sort_index().to_dict())

# 실험별 라벨 리본 (bat_class, tr_method, 길이순 정렬)
e2 = exp[exp.split != "test"].copy()
e2 = e2.sort_values(["bat_class", "tr", "tmax"])
fig, ax = plt.subplots(figsize=(15, 0.16 * len(e2) + 2))
for yi, (eid, r) in enumerate(e2.iterrows()):
    g = lab[lab.experiment_id == eid]
    t = g.relative_time_s.to_numpy(); l = g.label.to_numpy()
    cuts = np.r_[0, np.where(np.diff(l) != 0)[0] + 1, len(l)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        ax.broken_barh([(t[a], (t[b - 1] - t[a]) + 5)], (yi - 0.4, 0.8), color=LC[l[a]], lw=0)
ax.set_yticks(range(len(e2))); ax.set_yticklabels([f"bc{b} {cc:g}Ah tr{t} {'(val)' if s=='validation' else ''}" for b, cc, t, s in zip(e2.bat_class, e2.bat_cc, e2.tr, e2.split)], fontsize=5.5)
ax.set_xlabel("relative_time_s"); ax.set_title("Label timeline of every labeled experiment (color = label 1..6)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=LC[i], label=f"L{i}") for i in range(1, 7)], ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(); plt.show()
""")
code(r"""
# 단계별 시작 시각 / 지속 시간 (실험 단위) — 시간만으로 단계를 정할 수 없음을 확인
on = lab.groupby(["experiment_id", "label"]).relative_time_s.min().reset_index(name="onset")
on["bat_class"] = on.experiment_id.map(exp.bat_class)
dur = lab.groupby(["experiment_id", "label"]).size().reset_index(name="frames")
dur["bat_class"] = dur.experiment_id.map(exp.bat_class)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
sns.stripplot(data=on, x="label", y="onset", hue="bat_class", palette=BCC, ax=ax[0], size=3.5, alpha=.8, dodge=True); ax[0].set_title("Onset time of each label (per experiment)")
sns.boxplot(data=on, x="label", y="onset", color="#ddd", fliersize=0, ax=ax[0], zorder=0)
sns.stripplot(data=dur, x="label", y="frames", hue="bat_class", palette=BCC, ax=ax[1], size=3.5, alpha=.8, dodge=True, legend=False); ax[1].set_yscale("log"); ax[1].set_title("Frames spent in each label (log)")
piv = lab.assign(tbin=(lab.relative_time_s // 300) * 300).groupby(["tbin", "label"]).size().unstack(1).fillna(0)
(piv.div(piv.sum(1), axis=0)).plot.area(ax=ax[2], color=[LC[i] for i in range(1, 7)], legend=False); ax[2].set_title("Label mix vs elapsed time (300s bins)"); ax[2].set_xlim(0, 4000)
plt.tight_layout(); plt.show()
""")
md(r"""
> **설계 근거**: 모든 실험에서 라벨이 단조 비감소 → (a) *과거 예측의 누적 최댓값/과거 중앙값* 같은 **인과 후처리**가 이론적으로 유효,
> (b) 반면 단계 시작 시각의 편차가 매우 커서 `relative_time_s` 단독으로는 단계를 못 맞춘다(6장에서 time 피처 ablation).
> L6/L4 는 실험당 프레임이 적어 fold·validation 별 F1 변동이 크다.
""")

# ------------------------------------------------------------------ 4. 센서
md("## 4. 센서 EDA")
code(r"""
print("결측 비율 (split별):"); display(al.groupby("split")[["indu_t", "indu_t.1"]].apply(lambda d: d.isna().mean()).round(3))
print("indu_t 결측 vs tr_method:"); display(pd.crosstab(al.tr_method, al.indu_t.isna(), rownames=["tr_method"], colnames=["indu_t missing"]))
print("meas_a 고유값:", al.meas_a.unique(), "| t1+t2 오차 최대:", float((al.meas_t1 + al.meas_t2 - al.meas_t1_plus_t2).abs().max()))
""")
code(r"""
feats = ["meas_v", "meas_t1", "meas_t2", "meas_p", "thermal_t", "atmp_t", "meas_co", "meas_co2"]
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
for a, f in zip(axs.ravel(), feats):
    lo, hi = al[f].quantile([0.005, 0.995])
    for s in ["train", "validation", "test"]:
        sns.kdeplot(al.loc[(al.split == s) & al[f].between(lo, hi), f], ax=a, color=SPC[s], label=s, fill=True, alpha=.2, common_norm=False, warn_singular=False)
    a.set_title(f); a.set_xlabel("")
axs[0, 0].legend(); plt.suptitle("Sensor distribution by split (0.5–99.5 pct clip)", y=1.0); plt.tight_layout(); plt.show()
""")
code(r"""
# Adversarial validation: (train+val) vs test 를 센서만으로 구분할 수 있나? (실험 단위 GroupKFold)
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
def adv(cols, name):
    d = al[al.split != "validation"] if False else al
    y = (d.split == "test").astype(int).to_numpy(); X = d[cols]; g = d.experiment_id
    oof = np.zeros(len(d))
    for a, b in GroupKFold(5).split(X, y, g):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=15, verbose=-1, n_jobs=-1).fit(X.iloc[a], y[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    print(f"{name:34s} AUC(test vs rest) = {roc_auc_score(y, oof):.3f}")
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=15, verbose=-1, n_jobs=-1).fit(X, y)
S = C.SENSOR_BASE
adv(S, "sensors only"); m = adv(S + C.COND, "sensors + experiment conditions")
imp = pd.Series(m.booster_.feature_importance("gain"), index=S + C.COND).sort_values(ascending=False)
imp.head(10).plot.barh(figsize=(6, 3.2)).invert_yaxis(); plt.title("What separates test from train+val (gain)"); plt.show()
""")
md(r"""
> **결과**: 센서만으로 test vs train+val 을 구분하는 AUC 가 ≈0.5 → **센서 값 자체의 분포 이동은 거의 없다**. 즉 Train/Test 차이는 센서 스케일이 아니라
> *실험 구성(bat_class·용량 조합이 다름, test 는 11개뿐)* 과 *샘플링 간격* 에서 온다. 따라서 조건 코드(`bat_cc` 등)에 과의존하지 않고
> **조건 내 상대 변화(첫 관측 대비, delta, cummax)** 를 쓰는 설계가 안전하다(6장 ablation).
""")
code(r"""
# 라벨별 센서 분포
feats = ["meas_t1", "meas_t2", "thermal_t", "meas_v", "meas_p", "meas_co2", "meas_co", "meas_hf"]
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
for a, f in zip(axs.ravel(), feats):
    sns.boxplot(data=lab, x="label", y=f, palette=LCL, ax=a, fliersize=0.8, linewidth=.8)
    if f in ("meas_co2", "meas_co", "meas_hf", "meas_p"): a.set_yscale("symlog", linthresh=1)
    a.set_title(f)
plt.tight_layout(); plt.show()
""")
code(r"""
# bat_class 별로 보면 같은 라벨이어도 온도 스케일이 다르다 -> 절대 온도보다 '첫 관측 대비 상승량'이 필요
fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
for a, f in zip(axs, ["meas_t1", "thermal_t", "meas_t2"]):
    sns.boxplot(data=lab, x="label", y=f, hue="bat_class", palette=BCC, ax=a, fliersize=0.5, linewidth=.7); a.set_title(f + " by label x bat_class")
plt.tight_layout(); plt.show()
first = lab.sort_values("relative_time_s").groupby("experiment_id").first()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
sns.stripplot(data=first, x="bat_class", y="meas_t1", hue="tr_method", ax=ax[0]); ax[0].set_title("meas_t1 at t=0 (baseline differs per experiment)")
sns.stripplot(data=first, x="bat_class", y="meas_v", hue="bat_cc", ax=ax[1], palette="tab10"); ax[1].set_title("meas_v at t=0")
plt.tight_layout(); plt.show()
""")
code(r"""
# 대표 실험 시계열 (bat_class x tr_method 별 1개, 라벨 구간 음영)
combos = lab.groupby(["bat_class", "tr_method"]).experiment_id.unique()
fig, axs = plt.subplots(len(combos), 2, figsize=(16, 3.0 * len(combos)))
rng = np.random.default_rng(0)
for r, ((bc, trm), ids) in enumerate(combos.items()):
    ids = [i for i in ids if lab[lab.experiment_id == i].label.max() >= 5] or list(ids)
    g = lab[lab.experiment_id == rng.choice(ids)].sort_values("relative_time_s")
    for a, cols, ttl in [(axs[r, 0], ["meas_t1", "meas_t2", "thermal_t"], "temperatures"), (axs[r, 1], ["meas_co2", "meas_co", "meas_hf", "meas_p"], "gas / pressure (log1p)")]:
        for c in cols: a.plot(g.relative_time_s, np.log1p(g[c]) if "temp" not in ttl else g[c], label=c, lw=1)
        cuts = np.r_[0, np.where(np.diff(g.label.to_numpy()) != 0)[0] + 1, len(g)]
        for a0, b0 in zip(cuts[:-1], cuts[1:]):
            a.axvspan(g.relative_time_s.iloc[a0], g.relative_time_s.iloc[b0 - 1] + 5, color=LC[g.label.iloc[a0]], alpha=.22, lw=0)
        a.set_title(f"{BCN[bc]} / tr{trm} / {ttl}", fontsize=9); a.legend(fontsize=6, ncol=4)
plt.tight_layout(); plt.show()
""")
code(r"""
num = ["meas_v", "meas_t1", "meas_t2", "meas_p", "thermal_t", "atmp_t"] + C.GAS
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(lab[num + ["label"]].corr(method="spearman"), cmap="coolwarm", center=0, annot=True, fmt=".1f", annot_kws={"size": 6}, ax=ax); ax.set_title("Spearman correlation (train+val)")
plt.tight_layout(); plt.show()
""")

# ------------------------------------------------------------------ 5. 이미지
md("## 5. 이미지 EDA — 크기 지름길과 온도 인덱스 근사")
code(r"""
imf = pd.concat([C.load_image_cache(s).assign(split=s) for s in ["train", "validation", "test"]])
imf = imf.reset_index().rename(columns={"index": "sample_id"}).merge(al[["sample_id", "bat_class", "label", "experiment_id", "relative_time_s", "meas_t1", "thermal_t", "tr_method"]], on="sample_id")
imf["size"] = imf.size_w.astype(int).astype(str) + "x" + imf.size_h.astype(int).astype(str)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), sharex=True, sharey=True)
for a, s in zip(axs, ["train", "validation", "test"]):
    d = imf[imf.split == s]
    for bc in (1, 2, 3):
        dd = d[d.bat_class == bc]; a.scatter(dd.size_w + np.random.uniform(-2, 2, len(dd)), dd.size_h + np.random.uniform(-2, 2, len(dd)), s=3, alpha=.25, color=BCC[bc], label=BCN[bc])
    a.set_title(f"{s}: image size (w,h) by bat_class"); a.set_xlabel("width")
axs[0].set_ylabel("height"); axs[0].legend(markerscale=4); plt.tight_layout(); plt.show()
top = imf.groupby("size").agg(n=("sample_id", "size"), bc1=("bat_class", lambda s: (s == 1).mean()), bc2=("bat_class", lambda s: (s == 2).mean()), bc3=("bat_class", lambda s: (s == 3).mean())).sort_values("n", ascending=False).head(12)
print("가장 흔한 이미지 크기 12개와 bat_class 비율:"); display(top.round(3))
# 크기만으로 bat_class 를 맞추는 정확도 (크기별 최빈 클래스)
mode = imf.groupby("size").bat_class.agg(lambda s: s.mode().iloc[0]); print(f"크기 -> bat_class 최빈값 규칙 정확도: {(imf['size'].map(mode) == imf.bat_class).mean():.4f}")
""")
code(r"""
l = imf[imf.split != "test"].assign(label=lambda d: d.label.astype(int))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
sns.barplot(data=l, x="label", y="size_w", hue="bat_class", palette=BCC, ax=ax[0], errorbar=None); ax[0].set_title("Mean image width by label x bat_class")
mix = l.groupby(["label", "bat_class"]).size().unstack(1).fillna(0); (mix.div(mix.sum(1), axis=0)).plot.bar(stacked=True, ax=ax[1], color=[BCC[i] for i in (1, 2, 3)]); ax[1].set_title("bat_class share within each label")
sh = imf.groupby("split").bat_class.value_counts(normalize=True).unstack(0)[["train", "validation", "test"]]; sh.plot.bar(ax=ax[2], color=[SPC[c] for c in sh.columns]); ax[2].set_title("Row share by bat_class (image-size proxy)")
plt.tight_layout(); plt.show()
""")
md(r"""
> **핵심 위험**: 이미지 크기 → bat_class 가 거의 결정적이고, bat_class → 라벨 prior 가 다르다. 따라서 **크기 피처는 인코더/모델 입력에서 분리**하고
> (본 실험에서는 `size_*` 그룹) B0(크기만)·B0c(조건만) 점수를 하한선으로 둔 뒤, 크기를 넣었을 때/뺐을 때를 6장에서 비교한다.
""")
code(r"""
# 라벨 x bat_class 샘플 이미지 콜라주 (train, 같은 seed)
from PIL import Image
rng = np.random.default_rng(3)
trn = tr.merge(imf[["sample_id"]], on="sample_id")
fig, axs = plt.subplots(6, 6, figsize=(15, 14))
for li in range(1, 7):
    for bj, bc in enumerate((1, 2, 3)):
        pool = trn[(trn.label == li) & (trn.bat_class == bc)]
        for k in range(2):
            a = axs[li - 1, bj * 2 + k]; a.axis("off")
            if len(pool):
                r = pool.iloc[rng.integers(len(pool))]; a.imshow(Image.open(r.img_file).convert("RGB")); a.set_title(f"L{li} {BCN[bc][:4]} {r.size_w if hasattr(r,'size_w') else ''}", fontsize=7)
            else: a.text(.5, .5, f"no L{li}\n{BCN[bc]}", ha="center", va="center", fontsize=8)
plt.suptitle("Thermal frames by label (rows) x bat_class (column pairs)", y=1.0); plt.tight_layout(); plt.show()
""")
code(r"""
# 이미지에서 뽑은 '상대 온도 인덱스' 통계가 라벨을 구분하는가
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
for a, f in zip(axs.ravel(), ["img_mean", "img_p90", "img_p99", "img_hot67", "img_hot83", "img_white", "img_edge", "img_center_minus_all"]):
    sns.violinplot(data=l, x="label", y=f, palette=LCL, ax=a, cut=0, linewidth=.6, inner="quartile"); a.set_title(f)
plt.tight_layout(); plt.show()
fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
for a, bc in zip(axs, (1, 2, 3)):
    sns.lineplot(data=l[l.bat_class == bc], x="label", y="img_p99", hue="tr_method", ax=a, errorbar="sd"); a.set_title(f"img_p99 by label — {BCN[bc]}")
plt.tight_layout(); plt.show()
""")
code(r"""
# 이미지 통계 vs 센서 온도, 그리고 시간 흐름 (한 실험)
sm = l.sample(6000, random_state=0)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
sc = ax[0].scatter(sm.thermal_t, sm.img_p99, c=sm.label, cmap="viridis_r", s=4, alpha=.5); ax[0].set_xlabel("thermal_t (sensor)"); ax[0].set_ylabel("img_p99"); ax[0].set_title("thermal_t vs image hot-percentile"); plt.colorbar(sc, ax=ax[0], label="label")
ax[1].scatter(sm.meas_t1, sm.img_p99, c=sm.label, cmap="viridis_r", s=4, alpha=.5); ax[1].set_xscale("symlog"); ax[1].set_xlabel("meas_t1"); ax[1].set_title("meas_t1 vs img_p99")
eid = l.groupby("experiment_id").label.max().loc[lambda s: s == 6].index[0]; g = l[l.experiment_id == eid].sort_values("relative_time_s")
ax[2].plot(g.relative_time_s, g.img_p99, label="img_p99"); ax[2].plot(g.relative_time_s, g.img_hot67, label="img_hot67"); ax[2].plot(g.relative_time_s, g.img_white, label="img_white")
ax2 = ax[2].twinx(); ax2.plot(g.relative_time_s, g.thermal_t, color="k", lw=.8, label="thermal_t"); ax2.grid(False)
ax[2].legend(loc="upper left", fontsize=7); ax[2].set_title("One experiment: image stats vs thermal_t")
plt.tight_layout(); plt.show()
# Test 이미지 분포 이동
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for s in ["train", "validation", "test"]:
    sns.kdeplot(imf[imf.split == s].img_p99, ax=ax[0], color=SPC[s], label=s, fill=True, alpha=.2, common_norm=False); sns.kdeplot(imf[imf.split == s].img_mean, ax=ax[1], color=SPC[s], label=s, fill=True, alpha=.2, common_norm=False)
ax[0].legend(); ax[0].set_title("img_p99 by split"); ax[1].set_title("img_mean by split"); plt.tight_layout(); plt.show()
""")
md(r"""
> **관찰/설계**: 팔레트(자홍→…→빨강→흰색)를 hue 로 0~1 상대 온도 인덱스에 근사하면 L1→L6 으로 갈수록 고온 비율(`img_hot67/83`, `img_white`)이 증가한다.
> 다만 L4~L6 이 서로 가깝고 bat_class·거리에 따른 변동이 단계 차이만큼 크다 → 이미지 통계는 *센서 피처의 보조*로, 시간 변화량(causal delta)과 함께 사용한다.
""")

# ------------------------------------------------------------------ 6. 모델
md(r"""
## 6. 모델링 — 피처 그룹 ablation → 설정 선택 → 후처리 → 제출

`src/run_train.py` 가 (1) 피처 세트 ablation(B0/B0c 기준선 포함), (2) class_weight × 간격 증강 그리드, (3) 인과 후처리 선택을 **Validation Macro-F1** 기준으로 수행하고,
(4) 선택된 설정을 Train+Validation 으로 재학습해 `outputs/submission.csv` 를 만든다. (outputs 가 없으면 아래 셀이 자동 실행)
""")
code(r"""
import subprocess
OUT = C.OUT
if not (OUT / "submission.csv").exists():
    print("outputs/submission.csv 가 없어 학습을 먼저 실행합니다 ...")
    subprocess.run([sys.executable, "-u", "src/run_train.py"], check=True)
abl = pd.read_csv(OUT / "ablation.csv"); grid = pd.read_csv(OUT / "grid.csv"); post = pd.read_csv(OUT / "postproc.csv")
summ = json.loads((OUT / "summary.json").read_text()); print(json.dumps(summ, indent=1))
display(abl.round(4))
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(17, 4.8))
a2 = abl.set_index("feature_set")
w = 0.27; x = np.arange(len(a2))
ax[0].bar(x - w, a2.val_f1, w, label="Validation (19 exp)"); ax[0].bar(x, a2.valgap_f1, w, label="Validation, 5/10/15s-gap mix")
if "cv_f1_mean" in a2: ax[0].bar(x + w, a2.cv_f1_mean, w, yerr=a2.cv_f1_std, label="GroupKFold(5) train+val", capsize=2)
ax[0].set_xticks(x); ax[0].set_xticklabels(a2.index, rotation=35, ha="right", fontsize=8); ax[0].set_ylabel("Macro F1"); ax[0].set_title("Feature-set ablation"); ax[0].legend(fontsize=8)
sns.heatmap(a2[[f"f1_L{i}" for i in range(1, 7)]], annot=True, fmt=".2f", cmap="YlGnBu", ax=ax[1], vmin=0, vmax=1); ax[1].set_title("Validation per-class F1")
plt.tight_layout(); plt.show()
display(grid.round(4)); display(post.round(4))
""")
code(r"""
from sklearn.metrics import confusion_matrix, classification_report
vp = pd.read_parquet(OUT / "val_pred.parquet")
cm = confusion_matrix(vp.label, vp.pred_final, labels=C.LABELS)
fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt="d", cmap="Blues", ax=ax[0], xticklabels=C.LABELS, yticklabels=C.LABELS); ax[0].set_title("Validation confusion (counts, color=row-normalized)"); ax[0].set_xlabel("pred"); ax[0].set_ylabel("true")
from sklearn.metrics import f1_score
pe = vp.groupby("experiment_id").apply(lambda d: pd.Series({"f1": f1_score(d.label, d.pred_final, labels=C.LABELS, average="macro", zero_division=0), "acc": (d.label == d.pred_final).mean()}))
pe["bat_class"] = pe.index.map(exp.bat_class); sns.barplot(data=pe.reset_index(), x="experiment_id", y="acc", hue="bat_class", palette=BCC, ax=ax[1], dodge=False); ax[1].set_xticks([]); ax[1].set_title("Validation accuracy per experiment"); ax[1].set_xlabel("19 validation experiments")
imp = pd.read_csv(OUT / "feature_importance.csv").head(25); sns.barplot(data=imp, y="feature", x="gain", color="#4C78A8", ax=ax[2]); ax[2].set_title("Top-25 feature gain"); ax[2].tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.show()
print(classification_report(vp.label, vp.pred_final, digits=3, zero_division=0))
""")
code(r"""
# 검증 실험별 예측 타임라인 (위: 정답, 아래: 예측)
ids = list(vp.groupby("experiment_id").size().sort_values().index[::max(1, len(pe) // 8)])[:8]
fig, axs = plt.subplots(len(ids), 1, figsize=(14, 1.15 * len(ids) + 1), sharex=False)
for a, eid in zip(axs, ids):
    g = vp[vp.experiment_id == eid].sort_values("relative_time_s")
    a.scatter(g.relative_time_s, np.full(len(g), 1), c=[LC[v] for v in g.label], marker="|", s=250); a.scatter(g.relative_time_s, np.full(len(g), 0), c=[LC[v] for v in g.pred_final], marker="|", s=250)
    a.set_yticks([0, 1]); a.set_yticklabels(["pred", "true"], fontsize=7); a.set_title(f"bc{int(exp.loc[eid, 'bat_class'])} {exp.loc[eid, 'bat_cc']:g}Ah  acc={(g.label == g.pred_final).mean():.2f}", fontsize=8, loc="left"); a.set_ylim(-.7, 1.7)
plt.tight_layout(); plt.show()
""")
code(r"""
# 최종 submission 점검: 형식 + 예측 분포 + 테스트 실험별 타임라인
sub = pd.read_csv(OUT / "submission.csv", dtype={"sample_id": "string"}); ss = pd.read_csv(C.DATA_DIR / "sample_submission.csv", dtype={"sample_id": "string"})
assert list(sub.columns) == ["sample_id", "label"] and len(sub) == len(ss) and (sub.sample_id.values == ss.sample_id.values).all() and sub.label.between(1, 6).all()
print("submission OK:", sub.shape, "| 순서가 sample_submission 과 동일")
tp = te.merge(sub, on="sample_id", suffixes=("", "_pred")).sort_values(["experiment_id", "relative_time_s"])
fig, ax = plt.subplots(1, 2, figsize=(16, 4.2))
dd = pd.DataFrame({"train+val true": lab.label.value_counts(normalize=True), "test predicted": tp.label_pred.value_counts(normalize=True) if "label_pred" in tp else sub.label.value_counts(normalize=True)}).sort_index()
dd.plot.bar(ax=ax[0], color=["#4C78A8", "#54A24B"]); ax[0].set_title("Label share: true (train+val) vs predicted (test)")
for yi, (eid, g) in enumerate(tp.groupby("experiment_id")):
    col = "label_y" if "label_y" in g else ("label_pred" if "label_pred" in g else "label")
    ax[1].scatter(g.relative_time_s, np.full(len(g), yi), c=[LC[v] for v in g[col]], marker="|", s=150)
ax[1].set_yticks(range(tp.experiment_id.nunique())); ax[1].set_yticklabels([f"bc{b} {c:g}Ah tr{t}" for b, c, t in tp.groupby("experiment_id")[["bat_class", "bat_cc", "tr_method"]].first().itertuples(index=False)], fontsize=7); ax[1].set_title("Predicted label over time per test experiment")
plt.tight_layout(); plt.show()
""")

# ------------------------------------------------------------------ 6b. 센서 주도 + 이미지 게이팅
md(r"""
## 6-b. 센서가 못 맞추는 곳을 이미지로 보완 — 확신도 게이팅
이미지 피처를 전 프레임에 섞으면 오히려 하락(센서 상한 ≈ 0.92)한다는 선행 실험을 반영해, **센서 모델을 주(主)**로 두고
센서 예측이 {2,3(,4)} 이며 확신도(max prob) < τ 인 프레임에서만 이미지의 순서형 확률 P(y≥3), P(y≥4) 를 α 비율로 섞는다.
τ, α 는 Validation 이 아니라 **Train+Val 97개 실험 OOF** 로 정한다 (`src/run_gated.py`, outputs 없으면 자동 실행).
""")
code(r"""
if not (OUT / "gated_summary.json").exists():
    subprocess.run([sys.executable, "-u", "src/run_gated.py"], check=True)
gg = pd.read_csv(OUT / "gate_grid.csv"); ana = pd.read_csv(OUT / "gate_analysis.csv"); gs = json.loads((OUT / "gated_summary.json").read_text()); og = pd.read_parquet(OUT / "oof_gate.parquet")
print(json.dumps(gs, indent=1))
fig, axs = plt.subplots(1, 3, figsize=(18, 4.6))
for a, gset in zip(axs[:2], ["23", "234"]):
    pv = gg[gg.gset.astype(str) == gset].pivot(index="alpha", columns="tau", values="oof_f1"); sns.heatmap(pv, annot=True, fmt=".4f", cmap="YlGn", ax=a, cbar=False, annot_kws={"size": 7}); a.set_title(f"OOF macro-F1: gate on sensor-pred in {{{','.join(gset)}}}")
x = np.arange(6); axs[2].bar(x - .2, ana.sensor_recall, .4, label="sensor recall"); axs[2].set_xticks(x); axs[2].set_xticklabels([f"L{i}" for i in range(1, 7)]); axs[2].set_ylim(0, 1.05); axs[2].set_title("Sensor-only OOF recall per label")
for i, r in ana.iterrows(): axs[2].text(i - .2, r.sensor_recall + .01, f"wrong {r.sensor_wrong}\nrescued {r.rescued_by_gate}\nbroken {r.broken_by_gate}", ha="center", fontsize=6)
plt.tight_layout(); plt.show(); display(ana.round(3))
""")
code(r"""
# 센서 OOF 혼동행렬 vs 게이트 후, 그리고 센서 오답 프레임에서 이미지 확률의 분포
fig, axs = plt.subplots(1, 3, figsize=(18, 4.6))
for a, col, ttl in [(axs[0], "pred_sensor", "sensor-only OOF"), (axs[1], "pred_gated", "after gating")]:
    cm = confusion_matrix(og.label, og[col], labels=C.LABELS); sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt="d", cmap="Blues", ax=a, cbar=False, annot_kws={"size": 7}); a.set_title(f"{ttl}  F1={C.macro_f1(og.label, og[col]):.4f}"); a.set_xlabel("pred"); a.set_ylabel("true")
w = og[(og.label != og.pred_sensor) & og.pred_sensor.isin([2, 3, 4])]
sns.violinplot(data=w.assign(true=w.label.astype(str)), x="true", y="q3", ax=axs[2], cut=0, palette="viridis_r"); axs[2].axhline(.5, ls="--", c="k", lw=.7); axs[2].set_title("Image P(y>=3) on frames where sensor (pred 2-4) is wrong")
plt.tight_layout(); plt.show()
""")

# ------------------------------------------------------------------ 7. 해석 (결과 확인 후 보강)
md(r"""
## 7. 해석 요약 (EDA → 모델 설계 근거)
INTERPRETATION_PLACEHOLDER
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
path = ROOT / "experiment1.ipynb"
text = (ROOT / "src" / "interpretation.md").read_text(encoding="utf-8") if (ROOT / "src" / "interpretation.md").exists() else "(학습 실행 후 해석을 추가하세요)"
for c in nb.cells:
    if "INTERPRETATION_PLACEHOLDER" in c.source:
        c.source = c.source.replace("INTERPRETATION_PLACEHOLDER", text)
nbf.write(nb, path)
print("노트북 생성:", path)
