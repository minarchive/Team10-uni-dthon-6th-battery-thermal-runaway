"""Test 분포 맞춤(td) LightGBM + td 신경망 블렌드: OOF/가상 Test 를 계산하고 후보 CSV 를 만든다. 결과를 outputs/TD_BLEND.md 에 기록."""
import subprocess, sys
import numpy as np
from common import ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
from sklearn.metrics import f1_score
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0); fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values
P = ROOT / "outputs/preds"; f1 = lambda p: f1_score(y, p, labels=list(range(1, 7)), average="macro", zero_division=0)
base = np.load(P / "final_lgb_td_oof.npy"); out = [f"기준 final_lgb_td OOF {f1(F.postprocess(fr, base, 0.7)):.4f} (final_lgb 0.9184)", ""]
for run in ("all_hdtr_td", "all_hdtr_td_lr5e4", "all_hdtr_td_noimg"):
    f = P / f"ft_mm_{run}_oof_gated.npy"
    if not f.exists(): continue
    G = np.load(f); out.append(f"NN {run}: gated OOF {f1(F.postprocess(fr, G, 0.7)):.4f} | 블렌드 " + " ".join(f"w={w}: {f1(F.postprocess(fr, (1 - w) * base + w * G, 0.7)):.4f}" for w in (0.1, 0.2, 0.3, 0.5)))
    for w in (0.1, 0.3):
        r = subprocess.run([sys.executable, "scripts/make_nn_csv.py", run, "--w", str(w), "--base", "final_lgb_td"], cwd=ROOT, capture_output=True, text=True); out.append("  " + r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "  CSV 실패")
open(ROOT / "outputs/TD_BLEND.md", "w").write("```\n" + "\n".join(out) + "\n```\n"); print("\n".join(out))
