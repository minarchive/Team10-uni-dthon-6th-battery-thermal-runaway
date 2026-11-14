"""eda_v3.ipynb 생성: CPU 만으로(모델 재학습 없음) 하는 EDA 확장.
1 분할 구성(train/val/test x 형태 x 방식 x 용량 x SOC) + 순열 검정  2 기초 통계·이상치(왜도, IQR, Z)  3 실험 단위 기술자 분포(분할별, KS)
4 오류 분해 확장(조건 칸, 실험 내 위치, 오류 유형; 기존 OOF 재사용)  5 단계 지속시간(칸별)  6 변수 관계(전체 vs 실험 내 상관, 상호정보량, 지연 상관, PCA/군집)
7 저용량 모델(트리 아님): L2 vs L3 를 방식별로 로지스틱/단일 규칙/센서 OOF 와 비교
실행: python src/make_eda_v3.py
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
# EDA v3 — 분할 구성, 분포·이상치, 오류 분해 확장, 저용량 모델

모델 재학습 없이 CPU 로 하는 점검입니다(오류 분해는 기존 센서 OOF `outputs/oof_gate.parquet` 재사용).
분석용 통계(실험 전체 길이, 위치 등)는 모델 입력이 아닙니다.
""")

code(r"""
import sys, warnings, json
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
import common as C
sns.set_theme(style="whitegrid", context="notebook"); plt.rcParams.update({"figure.dpi": 105, "axes.titlesize": 10})
SPC = {"train": "#4C78A8", "validation": "#F58518", "test": "#54A24B"}
BCN = {1: "cyl", 2: "pouch", 3: "prism"}; TRN = {1: "heat", 2: "overchg"}
raw = C.load_all()
al = pd.concat([raw["train"], raw["validation"], raw["test"]], ignore_index=True).sort_values(["experiment_id", "relative_time_s"], kind="mergesort").reset_index(drop=True)
al["grp"] = al.bat_class.map(BCN) + "|" + al.tr_method.map(TRN)
lab = al[al.split != "test"].reset_index(drop=True)
def desc(g):
    t = g.relative_time_s.to_numpy(float); t1 = g.meas_t1.to_numpy(float); v = g.meas_v.to_numpy(float); cm = np.fmax.accumulate(t1)
    c = np.argmax(cm >= 45) if (cm >= 45).any() else None; vp = v[v > 0]
    return pd.Series(dict(split=g.split.iloc[0], grp=g.grp.iloc[0], bc=int(g.bat_class.iloc[0]), tr=int(g.tr_method.iloc[0]), cc=float(g.bat_cc.iloc[0]), soc=int(g.soc_status.iloc[0]),
        n=len(g), tmax=t.max(), dt=np.median(np.diff(t)) if len(t) > 1 else np.nan, t45=t[c] if c is not None else np.nan, t1_0=t1[0], t1max=t1.max(), thmax=g.thermal_t.max(),
        prise=g.meas_p.max() - g.meas_p.iloc[0], vmin=v.min() / (vp[0] if len(vp) else 1), gasmax=np.log1p(g[C.GAS].sum(axis=1)).max(), atmp0=g.atmp_t.iloc[0]))
ex = al.groupby("experiment_id").apply(desc)
print("실험", len(ex), "| split별:", ex.split.value_counts().to_dict())
""")

# ---------------------------------------------------------------- 1
md("## 1. 분할 구성 — train / validation / test 의 실험·형태·방식·용량·SOC")
code(r"""
order = ["train", "validation", "test"]
rows_cap = ex.assign(rows=np.where(ex.split == "test", ex.n, ex.n.clip(upper=500)))   # Test 와 같은 500행 상한 가정
fig, axs = plt.subplots(1, 3, figsize=(16, 3.6))
for a, s in zip(axs, order):
    t = ex[ex.split == s].groupby(["bc", "tr"]).size().unstack(fill_value=0).reindex(index=[1, 2, 3], columns=[1, 2]).fillna(0).astype(int)
    sns.heatmap(t, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a, xticklabels=["heat", "overchg"], yticklabels=["cyl", "pouch", "prism"]); a.set_title(f"{s}: experiments (n={int((ex.split == s).sum())})"); a.set_xlabel("method"); a.set_ylabel("shape")
plt.tight_layout(); plt.show()
comp = pd.DataFrame({s: ex[ex.split == s].groupby("grp").size() / (ex.split == s).sum() for s in order}).fillna(0)
rcomp = pd.DataFrame({s: rows_cap[rows_cap.split == s].groupby("grp").rows.sum() / rows_cap[rows_cap.split == s].rows.sum() for s in order}).fillna(0)
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
comp.T.plot.bar(stacked=True, ax=ax[0], colormap="tab20"); ax[0].set_title("Experiment share by split"); ax[0].legend(fontsize=7, bbox_to_anchor=(1, 1)); ax[0].set_xticklabels(order, rotation=0)
rcomp.T.plot.bar(stacked=True, ax=ax[1], colormap="tab20", legend=False); ax[1].set_title("Row share by split (rows capped at 500 per experiment)"); ax[1].set_xticklabels(order, rotation=0)
plt.tight_layout(); plt.show()
print("용량(Ah) x SOC x 방식, split 별 실험 수:"); display(ex.groupby(["bc", "cc", "soc", "tr", "split"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0))
""")
code(r"""
# 구성이 우연으로 설명되는지: 실험 라벨(split)을 무작위로 섞어 비교 (실험 수 보존, 5000회)
rng = np.random.default_rng(0); G = ex.grp.to_numpy(); gl = sorted(set(G)); gi = np.array([gl.index(g) for g in G])
def share(idx, w=None):
    c = np.bincount(gi[idx], weights=None if w is None else w[idx], minlength=len(gl)); return c / c.sum()
def dist(p, q): return float(((p - q) ** 2 / (p + q + 1e-9)).sum())
def perm_test(mask_a, mask_b, w=None, B=5000):
    ia, ib = np.where(mask_a)[0], np.where(mask_b)[0]; obs = dist(share(ia, w), share(ib, w)); pool = np.r_[ia, ib]; cnt = 0
    for _ in range(B):
        rng.shuffle(pool); cnt += dist(share(pool[:len(ia)], w), share(pool[len(ia):], w)) >= obs
    return obs, (cnt + 1) / (B + 1)
sp = ex.split.to_numpy(); w_rows = rows_cap.rows.to_numpy(float)
res = []
for nm, a, b in [("test vs train+val", sp == "test", sp != "test"), ("val vs train", sp == "validation", sp == "train")]:
    o1, p1 = perm_test(a, b); o2, p2 = perm_test(a, b, w_rows); res.append(dict(비교=nm, 실험비중_거리=o1, p실험=p1, 행비중_거리=o2, p행=p2))
display(pd.DataFrame(res).round(4))
# 라벨 비중: train vs validation (실험 단위 순열)
lab_ex = lab.groupby(["experiment_id", "label"]).size().unstack(fill_value=0).reindex(columns=range(1, 7), fill_value=0)
sp_l = ex.loc[lab_ex.index, "split"].to_numpy(); V = lab_ex.to_numpy(float)
def tv(ia, ib): a = V[ia].sum(0); b = V[ib].sum(0); return 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()
ia, ib = np.where(sp_l == "validation")[0], np.where(sp_l == "train")[0]; obs = tv(ia, ib); pool = np.r_[ia, ib]; cnt = 0
for _ in range(5000):
    rng.shuffle(pool); cnt += tv(pool[:len(ia)], pool[len(ia):]) >= obs
frame_p = stats.chi2_contingency(np.array([V[ia].sum(0), V[ib].sum(0)]))[1]
print(f"라벨 비중 train vs val: TV 거리 {obs:.3f} | 실험 단위 순열 p = {(cnt + 1) / 5001:.3f} | 프레임 단위 카이제곱 p = {frame_p:.2e}  (프레임 단위는 실험 군집을 무시해 과대 유의)")
""")
md(r"""
> **읽는 법**: 순열 p 가 크면 구성 차이가 실험 배정의 우연으로 설명된다는 뜻이다. Test 는 11개뿐이라 검정력이 낮아 'p 가 크다 = 같다' 는 아니다.
> 프레임 단위 카이제곱은 같은 실험의 프레임이 독립이 아니어서 차이를 과장한다(실험 단위 순열과 대비).
""")

# ---------------------------------------------------------------- 2
md("## 2. 기초 통계와 이상치 (EDA 절차 4·5)")
code(r"""
S = ["meas_v", "meas_t1", "meas_t2", "meas_p", "atmp_t", "thermal_t", "indu_t"] + C.GAS
def st(x):
    x = x.dropna().to_numpy(float); q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    z = np.abs((x - x.mean()) / (x.std() + 1e-12))
    return dict(n=len(x), mean=x.mean(), median=np.median(x), std=x.std(), skew=stats.skew(x), kurt=stats.kurtosis(x), zero=(x == 0).mean(), iqr_out=((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean(), z4=(z > 4).mean(), max=x.max())
tab = pd.DataFrame({c: st(lab[c]) for c in S}).T
display(tab.round(3).sort_values("skew", ascending=False))
sp_out = pd.DataFrame({s: {c: st(al[al.split == s][c])["iqr_out"] for c in S if al[al.split == s][c].notna().any()} for s in order}); print("IQR 이상치 비율, split 별:"); display(sp_out.round(3))
""")
code(r"""
fig, axs = plt.subplots(2, 4, figsize=(17, 6.4))
for a, c in zip(axs.ravel(), ["meas_v", "meas_t1", "meas_t2", "meas_p", "thermal_t", "atmp_t", "meas_co2", "meas_hf"]):
    for s in order:
        x = al.loc[al.split == s, c].dropna(); x = x[x > 0] if c in C.GAS else x; sns.kdeplot(np.log1p(x.clip(lower=0)), ax=a, color=SPC[s], label=s, fill=True, alpha=.2, common_norm=False, warn_singular=False)
    a.set_title(f"log1p({c})" + (" (>0 only)" if c in C.GAS else ""))
axs[0, 0].legend(); plt.tight_layout(); plt.show()
dead = al.groupby("experiment_id")[C.GAS].max().eq(0).groupby(ex.split).sum().T; print("가스 열이 실험 내내 0인 실험 수 (split별):"); display(dead[(dead > 0).any(axis=1)])
""")

# ---------------------------------------------------------------- 3
md("## 3. 실험 단위 기술자의 분할별 분포 (KS 검정)")
code(r"""
D = ["n", "tmax", "dt", "t45", "t1_0", "t1max", "thmax", "prise", "vmin", "gasmax", "atmp0"]
fig, axs = plt.subplots(2, 6, figsize=(19, 6))
for a, c in zip(axs.ravel(), D):
    sns.stripplot(data=ex, x="split", y=c, order=order, palette=SPC, ax=a, size=3.5, alpha=.7); sns.boxplot(data=ex, x="split", y=c, order=order, color="#ddd", fliersize=0, ax=a, zorder=0)
    a.set_title(c); a.set_xlabel("")
    if c in ("n", "tmax", "t45", "t1max", "thmax", "prise"): a.set_yscale("log")
for a in axs.ravel()[len(D):]: a.axis("off")
plt.tight_layout(); plt.show()
rows = []
for c in D:
    r = {"기술자": c}
    for nm, a, b in [("train vs test", "train", "test"), ("val vs test", "validation", "test"), ("train vs val", "train", "validation")]:
        xa, xb = ex.loc[ex.split == a, c].dropna(), ex.loc[ex.split == b, c].dropna(); r[nm + " KS p"] = stats.ks_2samp(xa, xb)[1] if len(xa) > 2 and len(xb) > 2 else np.nan
    rows.append(r)
ks = pd.DataFrame(rows).set_index("기술자"); display(ks.round(3))
print("KS p<0.05 개수:", int((ks < 0.05).sum().sum()), "/", ks.size, "(기술자 11개 x 3 비교 = 33회, 우연으로도 약 1.7회 기대)")
""")

# ---------------------------------------------------------------- 4
md(r"""
## 4. 오류 분해 확장 — 조건 칸, 실험 내 위치, 오류 유형
기존 센서 OOF(`oof_gate.parquet`, 전체 길이, cummax 후)를 재사용한다.
""")
code(r"""
oof = pd.read_parquet(C.OUT / "oof_gate.parquet")[["sample_id", "pred_sensor"]]
d = lab.merge(oof, on="sample_id"); d["pred"] = C.causal_cummax_pred(d, d.pred_sensor.to_numpy()); d["err"] = d.pred != d.label
d["pos"] = d.groupby("experiment_id").cumcount() / d.groupby("experiment_id").relative_time_s.transform("size")
print(f"OOF macro-F1 {C.macro_f1(d.label, d.pred):.4f} | 오류 {int(d.err.sum())}행 ({d.err.mean():.1%})")
cell = d.groupby(["bat_class", "bat_cc", "soc_status", "tr_method"]).agg(실험=("experiment_id", "nunique"), 행=("err", "size"), 오류=("err", "sum")).assign(오류율=lambda x: x.오류 / x.행, 오류비중=lambda x: x.오류 / x.오류.sum(), 행비중=lambda x: x.행 / x.행.sum())
cell["오류/행 비중"] = cell.오류비중 / cell.행비중
cell.index = [f"{BCN[a]} {int(b) if b == int(b) else b}Ah soc{c} {TRN[e]}" for a, b, c, e in cell.index]
display(cell.sort_values("오류비중", ascending=False).round(3))
""")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
d["pos_bin"] = pd.cut(d.pos, np.linspace(0, 1, 11), include_lowest=True, labels=[f"{i*10}-{i*10+10}%" for i in range(10)])
for tr in (1, 2): g = d[d.tr_method == tr].groupby("pos_bin").err.mean(); ax[0].plot(range(10), g.values, "o-", label=TRN[tr])
ax[0].set_xticks(range(10)); ax[0].set_xticklabels([f"{i*10}" for i in range(10)]); ax[0].set_xlabel("position in experiment (% of frames)"); ax[0].set_ylabel("error rate"); ax[0].legend(); ax[0].set_title("Where in the run errors occur")
pair = d[d.err].assign(pair=lambda x: x.label.astype(int).astype(str) + "->" + x.pred.astype(int).astype(str)).groupby(["tr_method", "pair"]).size().unstack(0).fillna(0).astype(int)
top = pair.assign(t=pair.sum(1)).sort_values("t", ascending=False).head(8).drop(columns="t"); top.columns = ["heat", "overchg"]; top.plot.barh(ax=ax[1]); ax[1].invert_yaxis(); ax[1].set_title("Error types (true->pred), top 8")
m = d.groupby(["bat_cc", "soc_status", "tr_method"]).err.mean().reset_index(); m["cell"] = m.bat_cc.astype(str) + "Ah s" + m.soc_status.astype(str) + " " + m.tr_method.map(TRN)
m = m.sort_values("err"); ax[2].barh(m.cell, m.err, color=["#D62728" if t == 2 else "#4C78A8" for t in m.tr_method]); ax[2].set_title("Error rate per capacity x SOC x method"); ax[2].tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 5
md("## 5. 단계 지속 시간 — 형태×방식 칸별")
code(r"""
dur = lab.groupby(["experiment_id", "label"]).size().reset_index(name="frames").merge(ex[["grp"]].reset_index(), on="experiment_id")
dur["sec"] = dur.frames * 5
fig, axs = plt.subplots(1, 6, figsize=(20, 4), sharey=False)
for L, a in zip(range(1, 7), axs):
    sns.boxplot(data=dur[dur.label == L], x="grp", y="sec", ax=a, fliersize=2); a.set_yscale("log"); a.set_title(f"L{L} duration (s)"); a.tick_params(axis="x", rotation=60, labelsize=7); a.set_xlabel("")
plt.tight_layout(); plt.show()
q = dur.groupby(["label", "grp"]).sec.agg(n="size", median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75)).round(0)
display(q.unstack(1)["median"])
for L in (2, 3):
    gs = [dur[(dur.label == L) & (dur.grp == g)].sec.values for g in sorted(dur.grp.unique())]; print(f"L{L} 지속 시간이 칸마다 다른가 (Kruskal-Wallis): H={stats.kruskal(*gs)[0]:.1f}, p={stats.kruskal(*gs)[1]:.2e}")
""")

# ---------------------------------------------------------------- 6
md("## 6. 변수 간 관계 (EDA 절차 6)")
code(r"""
X = ["meas_v", "meas_t1", "meas_t2", "meas_p", "thermal_t", "atmp_t"] + ["meas_co2", "meas_co", "meas_hf", "meas_hcn"]
sub = lab[X + ["experiment_id", "label"]].copy()
pooled = sub[X].corr(method="spearman"); within = (sub[X] - sub.groupby("experiment_id")[X].transform("mean")).corr(method="spearman")
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for a, m, t in [(ax[0], pooled, "pooled Spearman"), (ax[1], within, "within-experiment (experiment mean removed)")]:
    sns.heatmap(m, annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=a, cbar=False, annot_kws={"size": 7}); a.set_title(t)
plt.tight_layout(); plt.show()
from sklearn.feature_selection import mutual_info_classif
rs = lab.sample(20000, random_state=0); F = ["meas_v", "meas_t1", "meas_t2", "meas_p", "thermal_t", "atmp_t"] + C.GAS
rs["t1_rel"] = rs.meas_t1 - rs.groupby("experiment_id").meas_t1.transform("first"); rs["th_minus_amb"] = rs.thermal_t - rs.atmp_t; rs["gas_sum"] = np.log1p(rs[C.GAS].sum(axis=1))
F2 = F + ["t1_rel", "th_minus_amb", "gas_sum"]
mi = pd.Series(mutual_info_classif(rs[F2].fillna(0), rs.label, random_state=0), index=F2).sort_values()
mi_in = pd.Series({c: np.nanmean([mutual_info_classif(g[[c]].fillna(0), g.label, random_state=0)[0] for _, g in rs.groupby("grp") if g.label.nunique() > 1]) for c in F2}).reindex(mi.index)
fig, ax = plt.subplots(figsize=(8, 5)); y = np.arange(len(mi)); ax.barh(y - .2, mi.values, .4, label="all frames"); ax.barh(y + .2, mi_in.values, .4, label="within shape x method (mean)"); ax.set_yticks(y); ax.set_yticklabels(mi.index, fontsize=8); ax.legend(); ax.set_title("Mutual information with stage (no tree model)")
plt.tight_layout(); plt.show()
""")
code(r"""
# 지연 상관: 같은 실험 안에서 센서 1차 차분의 교차상관 (평균)
def xcorr(a, b, maxlag=6):
    out = {k: [] for k in range(-maxlag, maxlag + 1)}
    for _, g in lab.groupby("experiment_id"):
        x = g[a].diff().to_numpy()[1:]; y_ = g[b].diff().to_numpy()[1:]
        if len(x) < 40 or np.std(x) == 0 or np.std(y_) == 0: continue
        for k in out:
            xs, ys = (x[:len(x) - k], y_[k:]) if k >= 0 else (x[-k:], y_[:len(y_) + k])
            if np.std(xs) > 0 and np.std(ys) > 0: out[k].append(np.corrcoef(xs, ys)[0, 1])
    return pd.Series({k: np.nanmean(v) for k, v in out.items()})
fig, ax = plt.subplots(figsize=(8, 3.8))
for a, b in [("meas_t1", "thermal_t"), ("meas_t1", "meas_t2"), ("meas_v", "meas_p"), ("meas_t1", "meas_p")]: xcorr(a, b).plot(ax=ax, marker="o", label=f"d{a} vs d{b}")
ax.axvline(0, c="k", lw=.6); ax.set_xlabel("lag (frames, +: second signal later)"); ax.set_ylabel("mean correlation of first differences"); ax.legend(fontsize=7); ax.set_title("Lead/lag between signals (within experiment)")
plt.tight_layout(); plt.show()
""")
code(r"""
# 실험 기술자 PCA/군집: Test 실험이 학습 실험 구름 안에 있는가, 군집이 조건과 맞는가
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
Fd = ["tmax", "t45", "t1_0", "t1max", "thmax", "prise", "vmin", "gasmax"]
Z = ex[Fd].copy(); Z[["tmax", "t45", "t1max", "thmax", "prise"]] = np.log1p(Z[["tmax", "t45", "t1max", "thmax", "prise"]].clip(lower=0)); Z = StandardScaler().fit_transform(Z.fillna(Z.median()))
pc = PCA(2).fit(Z); P = pc.transform(Z)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for g, c in zip(sorted(ex.grp.unique()), sns.color_palette("tab10", 5)):
    m = (ex.grp == g).to_numpy(); ax[0].scatter(P[m & (ex.split != "test").to_numpy(), 0], P[m & (ex.split != "test").to_numpy(), 1], s=26, color=c, alpha=.6, label=g)
    ax[0].scatter(P[m & (ex.split == "test").to_numpy(), 0], P[m & (ex.split == "test").to_numpy(), 1], s=120, color=c, marker="*", edgecolor="k")
ax[0].legend(fontsize=7); ax[0].set_title(f"PCA of experiment descriptors (stars = test), var {pc.explained_variance_ratio_.sum():.0%}")
for s in order: m = (ex.split == s).to_numpy(); ax[1].scatter(P[m, 0], P[m, 1], s=30, color=SPC[s], alpha=.7, label=s)
ax[1].legend(); ax[1].set_title("same PCA colored by split")
plt.tight_layout(); plt.show()
cl = AgglomerativeClustering(5).fit_predict(Z); print(f"계층 군집(5) vs (형태,방식) 칸의 ARI = {adjusted_rand_score(ex.grp, cl):.2f}, 실루엣 {silhouette_score(Z, cl):.2f}  (ARI 가 낮으면 칸 라벨이 실험의 실제 유사성을 잘 설명하지 못함)")
""")

# ---------------------------------------------------------------- 7
md(r"""
## 7. 저용량 모델(트리 아님): L2 vs L3 를 방식별로
과충전에서 단순 규칙이 큰 모델보다 낫다는 관찰을 선형 모델로 확인한다. L2/L3 프레임만 대상으로 하고(실험 단위 5-fold), 센서 OOF(LightGBM 예측을 재사용, 새로 학습 안 함)와 같은 프레임에서 비교한다.
모든 입력은 인과적 누적 극값/현재 값이다.
""")
code(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler as SS
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
d = d.sort_values(["experiment_id", "relative_time_s"], kind="mergesort").reset_index(drop=True); g_ = d.groupby("experiment_id")
def cmax(x): return pd.Series(x).groupby(d.experiment_id.values).cummax().values
def cmin(x): return pd.Series(x).groupby(d.experiment_id.values).cummin().values
th, t1, t2, amb, p = (d[c].to_numpy(float) for c in ("thermal_t", "meas_t1", "meas_t2", "atmp_t", "meas_p"))
t1f = g_.meas_t1.transform("first").to_numpy(float); pf = g_.meas_p.transform("first").to_numpy(float)
def slope(x, w=30):
    o = np.full(len(d), np.nan); tt_ = d.relative_time_s.to_numpy(float)
    for _, ix in g_.indices.items():
        t = tt_[ix]; j = np.searchsorted(t, t - w, side="right") - 1; ok = j >= 0; v = np.full(len(ix), np.nan); v[ok] = (x[ix][ok] - x[ix][j[ok]]) / np.maximum(t[ok] - t[j[ok]], 1); o[ix] = v
    return o
FEAT = {"th-amb 누적최대": cmax(th - amb), "thermal 누적최대": cmax(th), "T1 상승 누적최대": cmax(t1 - t1f), "T2-T1 누적최소": cmin(t2 - t1), "thermal 기울기 누적최대": cmax(np.nan_to_num(slope(th), nan=0)), "압력 상승 누적최대": cmax(p - pf)}
FD = pd.DataFrame(FEAT); y23 = d.label.isin([2, 3]).to_numpy(); yb = (d.label.to_numpy() == 3).astype(int); sensor_b = (d.pred_sensor.to_numpy() >= 3).astype(int)
rows = []; oof_pred = {}
for tr in (1, 2):
    m = y23 & (d.tr_method.to_numpy() == tr); ii = np.where(m)[0]; grp_ = d.experiment_id.to_numpy()[ii]
    for name, cols in [("규칙(th-amb 임계, 폴드 내 튜닝)", None), ("로지스틱 1개 (th-amb)", ["th-amb 누적최대"]), ("로지스틱 3개", ["th-amb 누적최대", "T1 상승 누적최대", "T2-T1 누적최소"]), ("로지스틱 6개", list(FEAT))]:
        pr = np.zeros(len(ii), int)
        for a, b in GroupKFold(5, shuffle=True, random_state=0).split(ii, groups=grp_):
            if cols is None:
                v = FD["th-amb 누적최대"].to_numpy()[ii[a]]; grid = np.unique(np.nanpercentile(v, np.linspace(2, 98, 33))); best = max(grid, key=lambda g: f1_score(yb[ii[a]], (v >= g).astype(int), zero_division=0))
                pr[b] = (FD["th-amb 누적최대"].to_numpy()[ii[b]] >= best).astype(int)
            else:
                mdl = make_pipeline(SS(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=500)).fit(FD.iloc[ii[a]][cols], yb[ii[a]]); pr[b] = mdl.predict(FD.iloc[ii[b]][cols])
        oof_pred[(tr, name)] = (ii, pr)
        rows.append(dict(방식=TRN[tr], 모델=name, 실험=len(set(grp_)), L3_F1=f1_score(yb[ii], pr, zero_division=0), 균형정확도=balanced_accuracy_score(yb[ii], pr)))
    rows.append(dict(방식=TRN[tr], 모델="센서 LightGBM OOF (재사용)", 실험=len(set(grp_)), L3_F1=f1_score(yb[ii], sensor_b[ii], zero_division=0), 균형정확도=balanced_accuracy_score(yb[ii], sensor_b[ii])))
R = pd.DataFrame(rows); display(R.round(4))
# 실험 단위 부트스트랩: (로지스틱 3개 - 센서) 실험별 정확도 차이
rng = np.random.default_rng(0)
for tr in (1, 2):
    ii, pr = oof_pred[(tr, "로지스틱 3개")]; e = d.experiment_id.to_numpy()[ii]
    a = pd.Series(pr == yb[ii]).groupby(e).mean(); b = pd.Series(sensor_b[ii] == yb[ii]).groupby(e).mean(); dl = (a - b).to_numpy(); bs = [dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(2000)]
    print(f"{TRN[tr]}: 로지스틱3 − 센서 실험별 정확도 차이 평균 {dl.mean():+.4f}, 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}], 개선 {int((dl > 0).sum())}/악화 {int((dl < 0).sum())} 실험")
""")
md(r"""
## 8. 이상값과 센서 수준 — 999 같은 센티널 값이 있나, 가스가 실험 고유 수준을 담나
""")
code(r"""
num = [c for c in al.columns if al[c].dtype.kind in "fi" and c not in ("bat_class", "bat_cc", "soc_status", "tr_method", "relative_time_s")]
sent = [999, 9999, 99999, -999, -9999, -1, 65535, 32767, 999.9, 9999.9]
hits = {(c, s): int(np.isclose(al[c].to_numpy(float), s, atol=1e-6).sum()) for c in num for s in sent}; hits = {k: v for k, v in hits.items() if v}
print("센티널 후보(999, 9999, -999, 65535 등) 정확 일치:", hits if hits else "없음", "| 음수 값이 있는 열:", [c for c in num if (al[c] < 0).any()] or "없음")
print("참고: 값이 정확히 99.0 인 행(T1 17, T2 27 등)은 연속 온도값의 우연 일치이고 센티널이 아님. 극단값:", al[["meas_t1", "meas_t2", "thermal_t", "atmp_t", "meas_v"]].max().round(1).to_dict())
l1 = lab[lab.label == 1]
def eta2(df, c):
    x = df[c].astype(float); m = x.mean(); gm = df.groupby("experiment_id")[c].transform("mean"); return ((gm - m) ** 2).sum() / ((x - m) ** 2).sum()
cols = C.GAS + ["meas_t1", "meas_t2", "thermal_t", "meas_v", "meas_p", "atmp_t"]
e2 = pd.Series({c: eta2(l1, c) for c in cols}).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
ax[0].barh(e2.index, e2.values, color=["#D62728" if c in C.GAS else "#4C78A8" for c in e2.index]); ax[0].set_xlabel("share of variance explained by experiment identity (L1 frames only)")
ax[0].set_title("Healthy-state (L1) level is experiment-specific? red = gases")
g = lab.assign(gs=np.log1p(lab[C.GAS].sum(axis=1))); b = g[g.label == 1].groupby("experiment_id").gs.mean(); a3 = g[g.label == 3].groupby("experiment_id").gs.mean(); j = pd.concat([b, a3], axis=1, keys=["L1", "L3"]).dropna()
ax[1].scatter(j.L1, j.L3, s=22, alpha=.7); ax[1].plot([0, j.max().max()], [0, j.max().max()], "k--", lw=.6); ax[1].set_xlabel("log1p(gas sum) mean in L1, per experiment"); ax[1].set_ylabel("same in L3")
ax[1].set_title(f"Between-experiment SD at L1 = {j.L1.std():.2f}, mean change L1->L3 = {(j.L3 - j.L1).mean():.2f}")
plt.tight_layout(); plt.show()
sp = tot = 0
for e, x in l1.groupby("experiment_id"):
    v = np.log1p(x[C.GAS].sum(axis=1).to_numpy()); med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-6; sp += int((np.abs(v - med) > 6 * mad * 1.4826).sum()); tot += len(v)
print(f"L1 프레임 중 실험 내 6-MAD 이탈 가스 프레임 {sp}/{tot} ({sp / tot:.2%}) -> 스파이크가 아니라 '실험마다 다른 수준'이 문제")
""")
code(r"""
gl = al[al.atmp_t > 60]
print(f"atmp_t>60: {len(gl)}행, 실험 {gl.experiment_id.nunique()}개, split {gl.split.value_counts().to_dict()}, 형태 {gl.bat_class.map(BCN).value_counts().to_dict()}, Test 최대 atmp_t {al[al.split == 'test'].atmp_t.max()}")
print("라벨 분포:", gl.label.value_counts().sort_index().to_dict())
rows = []
for e, x in gl.groupby("experiment_id"):
    full = al[al.experiment_id == e].reset_index(drop=True); idx = np.where(full.atmp_t > 60)[0]
    rows.append(dict(실험=e[:8], 형태=BCN[int(x.bat_class.iloc[0])], 방식=TRN[int(x.tr_method.iloc[0])], 행=len(idx), 구간수=int((np.diff(idx) > 1).sum() + 1), 시작위치=idx[0] / len(full), T1중앙=x.meas_t1.median(), 라벨=sorted(set(int(v) for v in x.label.dropna()))))
display(pd.DataFrame(rows).round(2))
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.hist([r["시작위치"] for r in rows], bins=np.linspace(.9, 1, 11), color="#D62728"); ax.set_xlabel("position in experiment where atmp_t first exceeds 60C"); ax.set_ylabel("# experiments"); ax.set_title("atmp_t>60 rows all sit at the very end of a run (the runaway itself)")
plt.tight_layout(); plt.show()
""")
md(r"""
## 결론
EDA3_CONCLUSIONS_PLACEHOLDER
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
p = ROOT / "src" / "eda_v3_conclusions.md"
text = p.read_text(encoding="utf-8") if p.exists() else "(실행 후 결론을 추가하세요)"
for c in nb.cells:
    if "EDA3_CONCLUSIONS_PLACEHOLDER" in c.source:
        c.source = c.source.replace("EDA3_CONCLUSIONS_PLACEHOLDER", text)
nbf.write(nb, ROOT / "eda_v3.ipynb")
print("생성:", ROOT / "eda_v3.ipynb")
