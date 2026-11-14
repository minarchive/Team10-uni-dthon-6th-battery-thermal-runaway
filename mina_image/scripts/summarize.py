"""probe_results.csv -> 인코더별 최고 레이어 표 + 레이어 스윕 그림(outputs/figures/layer_sweep.png)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

r = pd.read_csv(ROOT / "outputs" / "logs" / "probe_results.csv").drop_duplicates(["encoder", "layer"], keep="last")
base = r[r.layer == "-"]
r = r[r.layer != "-"]
print(base[["encoder", "f1_mean", "f1_std", "oof_f1"]].round(3).to_string(index=False))
best = r.loc[r.groupby("encoder").f1_mean.idxmax()][["encoder", "layer", "f1_mean", "f1_std", "oof_f1", "adj_acc", "stage_mae", "gap_train_minus_val"]]
print(best.round(3).to_string(index=False))
best.to_csv(ROOT / "outputs" / "logs" / "best_layer_by_encoder.csv", index=False)
encs = sorted(r.encoder.unique())
fig, axs = plt.subplots(len(encs), 1, figsize=(10, 2.6 * len(encs)), squeeze=False)
for ax, e in zip(axs[:, 0], encs):
    s = r[r.encoder == e]
    for kind in sorted(set(l.split("_")[1] for l in s.layer)):
        t = s[s.layer.str.endswith("_" + kind)].sort_values("layer")
        ax.plot(t.layer, t.f1_mean, marker="o", label=kind)
    for _, b in base.iterrows():
        ax.axhline(b.f1_mean, ls="--", lw=0.8, color="gray")
    ax.set_title(e)
    ax.set_ylabel("Macro F1 (5-fold)")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "figures" / "layer_sweep.png", dpi=130)
print("saved layer_sweep.png")
