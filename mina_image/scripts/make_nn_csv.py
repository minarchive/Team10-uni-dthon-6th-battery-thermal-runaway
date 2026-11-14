"""신경망 Test 확률(ft_mm_<run>_test.npy, fold 평균)에 L1 규칙(T1 누적최대<45℃ -> L1 고정) + 확신 cummax 후처리를 적용해 제출 CSV 생성.
--w > 0 이면 final_lgb_test.npy(과충전 규칙 포함 LightGBM 확률)와 (1-w):w 로 블렌드. 인과성: 같은 실험의 현재·과거 관측만 사용.
사용: python scripts/make_nn_csv.py all_hdtr --w 0.3 -> outputs/submissions/mm_<run>[_blendXX].csv"""
import argparse, sys
import numpy as np, pandas as pd
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--w", type=float, default=0.0); ap.add_argument("--base", default="final_lgb"); a = ap.parse_args()
L = list(range(1, 7)); tb = table(); te = np.where(tb.label.values == -1)[0]
raw = pd.read_csv(DATA / "test" / "test.csv", dtype={"sample_id": "string", "experiment_id": "string"}); raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
ft = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values[te]].reset_index(drop=True)          # te 행 순서(NN·final_lgb 와 동일)
P = np.load(ROOT / "outputs/preds" / f"ft_mm_{a.run}_test.npy"); assert len(P) == len(ft)
cm = np.zeros(len(ft)); t = ft.relative_time_s.values.astype(float); t1 = ft.meas_t1.values.astype(float)
for e, ix in pd.Series(np.arange(len(ft))).groupby(ft.experiment_id.values).indices.items():
    ix = ix[np.argsort(t[ix], kind="stable")]; cm[ix] = np.maximum.accumulate(t1[ix])
r = cm >= 45; G = P.copy(); G[~r] = 0; G[~r, 0] = 1.0; G[r, 0] = 0; s = G[r].sum(1, keepdims=True); G[r] = np.where(s > 0, G[r] / np.where(s > 0, s, 1), 0.2); G[r, 0] = 0
if a.w > 0: G = (1 - a.w) * np.load(ROOT / "outputs/preds" / f"{a.base}_test.npy") + a.w * G
lab = F.postprocess(ft[["sample_id", "experiment_id", "relative_time_s"]], G, 0.7)
order = pd.read_csv(DATA / "sample_submission.csv", dtype={"sample_id": "string"})["sample_id"].astype(str).tolist()
sub = pd.DataFrame({"sample_id": order, "label": pd.Series(lab.astype(int), index=ft.sample_id.values).loc[order].values})
assert len(sub) == 3451 and sub.label.isin(L).all() and sub.sample_id.is_unique
name = f"mm_{a.run}" + (f"_blend{int(round(a.w * 100)):02d}" + ("" if a.base == "final_lgb" else "_" + a.base.replace("final_lgb_", "")) if a.w > 0 else "_gated"); out = ROOT / "outputs/submissions" / f"{name}.csv"; sub.to_csv(out, index=False)
ref = pd.read_csv(ROOT / "outputs/submissions/final_lgb.csv", dtype={"sample_id": "string"}).set_index("sample_id").label
print(f"저장 {out.name} | 분포 {sub.label.value_counts(normalize=True).sort_index().round(3).to_dict()} | final_lgb 와 일치율 {(ref.loc[sub.sample_id.values].values == sub.label.values).mean():.4f}")
