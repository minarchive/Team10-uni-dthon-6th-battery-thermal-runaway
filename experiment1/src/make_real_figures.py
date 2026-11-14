"""실제 열화상 프레임 + 시계열 그림 (outputs/evidence/real_*.png). 데이터 파생물 -> 커밋 금지.
1) 단계별 프레임 콜라주  2) 실험 1개: 단계 시작 프레임 + 시계열 + 라벨 띠(형태별 3개)
3) 이미지 가공 단계(원본/흑백/온도 인덱스/배경 마스크/64x64)  4) 카메라 구도(이미지 크기별)  5) 절대 vs 첫 프레임 대비 상대(이미지 통계·센서, 전 실험)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import common as C
import events as E

OUT = C.OUT / "evidence"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="white")
LC = dict(zip(range(1, 7), sns.color_palette("viridis_r", 6)))
BCN = {1: "cylindrical", 2: "pouch", 3: "prismatic"}
BCC = {1: "#9467BD", 2: "#2CA02C", 3: "#D62728"}

lab = pd.concat([C.load_split(s) for s in ("train", "validation")]).sort_values(["experiment_id", "relative_time_s"], kind="mergesort").reset_index(drop=True)
cache = pd.concat([C.load_image_cache(s) for s in ("train", "validation")])
lab = lab.merge(cache.reset_index().rename(columns={"index": "sample_id"}), on="sample_id", how="left")
ev = E.event_features(lab[[c for c in lab.columns if c in C.load_split("validation").columns]])
lab["v_ratio"] = ev.ev_v_ratio.values; lab["p_rise"] = ev.ev_p_rise.values; lab["gas_sum"] = np.log1p(lab[C.GAS].sum(axis=1))
lab["t1_rise"] = lab.meas_t1 - lab.groupby("experiment_id").meas_t1.transform("first")
lab["fidx"] = lab.groupby("experiment_id").cumcount()
exp = lab.groupby("experiment_id").agg(bc=("bat_class", "first"), tr=("tr_method", "first"), n=("label", "size"), nl=("label", "nunique"), lmax=("label", "max"))
img = lambda p: Image.open(p).convert("RGB")

# ---------------- 1) 단계별 콜라주
rng = np.random.default_rng(3)
fig, axs = plt.subplots(6, 6, figsize=(13, 12))
for li in range(1, 7):
    for bj, bc in enumerate((1, 2, 3)):
        pool = lab[(lab.label == li) & (lab.bat_class == bc)]
        exps = pool.experiment_id.unique()
        for k in range(2):
            a = axs[li - 1, bj * 2 + k]; a.axis("off")
            if len(exps):
                e = exps[rng.integers(len(exps))]; g = pool[pool.experiment_id == e]; r = g.iloc[len(g) // 2]
                a.imshow(img(r.img_file)); a.set_title(f"L{li} | {BCN[bc][:4]} | t={int(r.relative_time_s)}s", fontsize=7)
            else:
                a.text(.5, .5, f"no L{li}\n{BCN[bc]}", ha="center", va="center", fontsize=8)
plt.suptitle("Thermal frames by stage (rows) and battery shape (column pairs), train+validation", y=0.995, fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "real_stage_collage.png", dpi=110); plt.close()

# ---------------- 2) 실험 1개: 프레임 + 시계열
def pick(bc):
    c = exp[(exp.bc == bc) & (exp.tr == 1) & (exp.nl == 6)]
    return (c.n - c.n.median()).abs().sort_values().index[0]
for bc in (1, 2, 3):
    eid = pick(bc); g = lab[lab.experiment_id == eid].reset_index(drop=True); t = g.relative_time_s.to_numpy()
    on = g.groupby("label").head(1)
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(5, 6, height_ratios=[1.25, 0.12, 1, 1, 1], hspace=0.35)
    for i, (_, r) in enumerate(on.iterrows()):
        a = fig.add_subplot(gs[0, i]); a.imshow(img(r.img_file)); a.axis("off"); a.set_title(f"L{int(r.label)} onset  t={int(r.relative_time_s)}s\nT1={r.meas_t1:.0f}C", fontsize=8, color=LC[int(r.label)])
        for s in a.spines.values(): s.set_visible(False)
    axs = [fig.add_subplot(gs[2 + k, :]) for k in range(3)]
    for a in axs: a.margins(x=0)
    axs[0].plot(t, g.meas_t1, label="T1 (surface)", lw=1.4); axs[0].plot(t, g.meas_t2, label="T2 (positive terminal)", lw=1.2); axs[0].plot(t, g.thermal_t, label="thermal_t", lw=1.2); axs[0].set_ylabel("temperature (C)"); axs[0].legend(loc="upper left", fontsize=8, ncol=3)
    axs[1].plot(t, g.v_ratio.clip(0, 2), label="V / V0", lw=1.3, color="#6f42c1"); a2 = axs[1].twinx(); a2.plot(t, g.p_rise, color="#e8590c", lw=1.1, label="pressure rise"); a2.grid(False)
    axs[1].set_ylabel("V / V0"); a2.set_ylabel("pressure rise"); axs[1].legend(loc="upper left", fontsize=8); a2.legend(loc="upper right", fontsize=8)
    axs[2].plot(t, g.gas_sum, label="log1p(sum of gases)", lw=1.3, color="#2b8a3e"); axs[2].plot(t, g.img_p90 * g.gas_sum.max(), label="image p90 temp-index (scaled)", lw=1.0, color="#868e96", ls="--")
    axs[2].set_ylabel("gas / image"); axs[2].set_xlabel("relative_time_s"); axs[2].legend(loc="upper left", fontsize=8)
    for a in axs:
        a.grid(alpha=.25)
        for _, r in on.iterrows():
            a.axvline(r.relative_time_s, color=LC[int(r.label)], lw=.8, ls=":")
        cuts = np.r_[0, np.where(np.diff(g.label.to_numpy()) != 0)[0] + 1, len(g)]
        for a0, b0 in zip(cuts[:-1], cuts[1:]):
            a.axvspan(t[a0], t[b0 - 1] + 5, color=LC[int(g.label[a0])], alpha=.16, lw=0)
    fig.suptitle(f"One experiment: {BCN[bc]} cell, {g.bat_cc.iloc[0]:g} Ah, heating, {len(g)} frames. Shading = label (light L1 ... dark L6), frames = first frame of each label", y=0.985, fontsize=10.5)
    plt.savefig(OUT / f"real_experiment_{bc}.png", dpi=105, bbox_inches="tight"); plt.close()

# ---------------- 3) 이미지 가공 단계 (실제 프레임)
eid = pick(3); g = lab[lab.experiment_id == eid]
frames = [g[g.label == 2].iloc[len(g[g.label == 2]) // 2], g[g.label == 4].iloc[len(g[g.label == 4]) // 2]]
fig, axs = plt.subplots(2, 6, figsize=(14, 5.4))
for i, r in enumerate(frames):
    im = img(r.img_file); a_ = np.asarray(im, dtype=np.float32) / 255.0
    gray = 0.299 * a_[..., 0] + 0.587 * a_[..., 1] + 0.114 * a_[..., 2]
    idx, white, valid = C._temp_index(a_); body = (idx > 0.12) & valid
    small = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0; i2, w2, v2 = C._temp_index(small)
    panels = [(a_, None, "1. original RGB"), (gray, "gray", "2. grayscale (baseline stats)"), (idx, "magma", "3. temperature index 0-1"),
              (np.where(body, idx, np.nan), "magma", f"4. body mask (frac {body.mean():.2f})"), (idx - 0, "magma", None), (i2, "magma", "5. 64x64 resize, temp index")]
    for j, (arr, cm, ttl) in enumerate(panels):
        a = axs[i, j]; a.axis("off")
        if j == 4:
            hv = idx[body] if body.any() else idx.ravel()
            a.hist(hv, bins=30, color="#3b5bdb"); a.axis("on"); a.set_yticks([]); a.set_title(f"histogram (body)\np50={np.percentile(hv, 50):.2f} p90={np.percentile(hv, 90):.2f} p99={np.percentile(hv, 99):.2f}", fontsize=8); continue
        a.imshow(arr, cmap=cm, vmin=0 if cm else None, vmax=1 if cm else None)
        a.set_title((ttl + f"\nL{int(r.label)}  T1={r.meas_t1:.0f}C") if j == 0 else ttl, fontsize=8)
plt.suptitle("Image processing steps on real frames (same experiment, stage L2 above, stage L4 below)", y=0.995, fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT / "real_processing.png", dpi=110); plt.close()

# ---------------- 4) 카메라 구도
sizes = lab.groupby(["size_w", "size_h"]).agg(n=("label", "size"), bc=("bat_class", lambda s: s.mode().iloc[0]), path=("img_file", "first")).sort_values("n", ascending=False).reset_index()
fig, axs = plt.subplots(2, 7, figsize=(14, 4.4))
for a, (_, r) in zip(axs.ravel(), sizes.iterrows()):
    a.imshow(img(r.path)); a.axis("off"); a.set_title(f"{int(r.size_w)}x{int(r.size_h)}\n{BCN[int(r.bc)][:4]}  n={int(r.n)}", fontsize=8, color=BCC[int(r.bc)])
for a in axs.ravel()[len(sizes):]: a.axis("off")
plt.suptitle("Camera framing differs by experiment: image size is a proxy for battery shape (one frame per size)", y=0.995, fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT / "real_framing.png", dpi=110); plt.close()

# ---------------- 5) 절대 vs 첫 프레임 대비 상대 (전 실험)
sm = lambda s: s.rolling(7, min_periods=1, center=False).median()   # 과거 방향 rolling
fig, axs = plt.subplots(2, 2, figsize=(13, 7.4), sharex=True)
for eid_, g in lab.groupby("experiment_id"):
    bc = int(g.bat_class.iloc[0]); tc = g.loc[g.meas_t1.cummax() >= 45, "relative_time_s"].min()
    if np.isnan(tc): continue
    ta = (g.relative_time_s - tc).to_numpy(); m = (ta >= -300) & (ta <= 2500)
    for r_, (col, ttl) in enumerate([("img_p90", "image p90 temp-index"), ("thermal_t", "thermal_t sensor (C)")]):
        y = sm(g[col]); axs[r_, 0].plot(ta[m], y[m], color=BCC[bc], alpha=.35, lw=.8); axs[r_, 1].plot(ta[m], (y - y.iloc[0])[m], color=BCC[bc], alpha=.35, lw=.8)
        axs[r_, 0].set_ylabel(ttl)
axs[0, 0].set_title("absolute value"); axs[0, 1].set_title("minus the experiment's first frame (past only)")
for a in axs[1]: a.set_xlabel("time since T1 >= 45C (s)")
axs[0, 0].legend(handles=[plt.Line2D([], [], color=BCC[b], label=BCN[b]) for b in (1, 2, 3)], fontsize=8)
for a in axs.ravel(): a.grid(alpha=.25)
plt.suptitle("Absolute vs minus-first-frame, all 97 experiments: subtracting removes the starting offset (it is 0 by construction), but image trajectories still fan out", y=0.995, fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "real_relative.png", dpi=110); plt.close()
print("done", sorted(p.name for p in OUT.glob("real_*.png")))
