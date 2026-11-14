"""신경망(및 센서) OOF 확률에 규칙 제약만 후처리로 적용해 점수 변화 확인(재학습 없음). 규칙: T1 누적최댓값 < 45°C 이면 L1 확정, >= 45°C 이면 L1 불가(L2~L6 중 선택). 과거 값(누적)만 사용."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=["sample_id", "experiment_id", "relative_time_s", "meas_t1"]) for s in ["train", "validation"]])
raw["sample_id"] = raw.sample_id.astype(str); raw = raw.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index(); raw["experiment_id"] = raw.experiment_id.astype(str)
eid = raw.experiment_id.values; t = raw.relative_time_s.values.astype(float); t1 = raw.meas_t1.values.astype(float); cm = np.zeros(len(y))
for e, ix in pd.Series(np.arange(len(y))).groupby(eid).indices.items():
    ix = ix[np.argsort(t[ix], kind="stable")]; cm[ix] = np.maximum.accumulate(t1[ix])
reached = cm >= 45
f1 = lambda p: f1_score(y, p, labels=L, average="macro", zero_division=0)
def gate(P):
    G = P.copy(); G[~reached] = 0; G[~reached, 0] = 1.0; G[reached, 0] = 0
    s = G[reached].sum(1, keepdims=True); G[reached] = np.where(s > 0, G[reached] / np.where(s > 0, s, 1), 1 / 5)
    G[reached, 0] = 0; G[reached] = G[reached] / np.where(G[reached].sum(1, keepdims=True) > 0, G[reached].sum(1, keepdims=True), 1); return G
lg = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy")
print(f"{'모델':18s} {'기준':>7s} {'규칙 후처리':>10s} {'Δ':>8s} | 이긴 fold | L1<->L2/L1->L3 오류 전→후 | 단계별(후)")
for nm in ["sensor_lgb", "ft_mm_fusion", "ft_mm_noimg", "ft_mm_tab100", "ft_mm_tab100_noimg", "ft_mm_tab100_img4", "ft_mm_mm", "ft_mm_img4"]:
    f = ROOT / "outputs/preds" / (f"{nm}_oof.npy"); 
    if not f.exists(): continue
    P = np.load(f); pb = F.postprocess(fr, P, 0.7); pg = F.postprocess(fr, gate(P), 0.7)
    fs = lambda p: np.array([f1_score(y[fold == k], p[fold == k], labels=L, average="macro") for k in range(5)])
    e1 = lambda p: int((((y == 1) & (p == 2)) | ((y == 2) & (p == 1)) | ((y == 1) & (p == 3)) | ((y == 3) & (p == 1))).sum())
    print(f"{nm:18s} {f1(pb):7.4f} {f1(pg):10.4f} {f1(pg) - f1(pb):+8.4f} | {int((fs(pg) > fs(pb)).sum())}/5 | {e1(pb):5d} -> {e1(pg):5d} | {np.round(f1_score(y, pg, labels=L, average=None), 3).tolist()}")
    if nm.startswith("ft_mm"):
        for w in (0.3,):
            pbl = F.postprocess(fr, (1 - w) * lg + w * gate(P), 0.7); print(f"{'':18s} 규칙 후처리 후 LightGBM 블렌드 w={w}: {f1(pbl):.4f} ({f1(pbl) - f1(F.postprocess(fr, lg, 0.7)):+.4f})")
