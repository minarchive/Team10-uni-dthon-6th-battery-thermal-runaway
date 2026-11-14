"""오류 분해: (형태 x 방식) x 오류 유형 x 모델(센서 LightGBM / 신경망 tab100), L2->L3 전환 시각 오차(일찍/늦게)."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values
meta = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=["experiment_id", "bat_class", "tr_method", "bat_cc"]).drop_duplicates("experiment_id") for s in ["train", "validation"]]); meta["experiment_id"] = meta.experiment_id.astype(str); m = meta.set_index("experiment_id")
fr["cls"] = fr.experiment_id.map(m.bat_class).values; fr["tr"] = fr.experiment_id.map(m.tr_method).values
P = {"센서 LightGBM": np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy"), "신경망 tab100": np.load(ROOT / "outputs/preds/ft_mm_tab100_oof.npy")}
NAME = {1: "원통", 2: "파우치", 3: "각형"}; TR = {1: "가열", 2: "과충전"}
t = fr.relative_time_s.values.astype(float); eid = fr.experiment_id.values
idx = {e: np.where(eid == e)[0][np.argsort(t[eid == e], kind="stable")] for e in np.unique(eid)}
for mn, Pm in P.items():
    p = F.postprocess(fr, Pm, 0.7); err = p != y
    print(f"\n==================== {mn} (OOF F1 {f1_score(y, p, labels=L, average='macro'):.4f}, 오류 {err.sum()}행) ====================")
    print("(형태, 방식)별: 행 / 오류율 / 전체 오류 비중 | L2<->L3 오류 비중(그 그룹 안) | L4~L6 사이 오류 비중 | 오류 유형 상위")
    for (c, tr), g in fr.groupby(["cls", "tr"]):
        ii = g.index.values; e_ = err[ii]; yy, pp = y[ii], p[ii]
        l23 = ((yy == 2) & (pp == 3)) | ((yy == 3) & (pp == 2)); late = ((yy >= 4) & (pp >= 4) & (yy != pp)) | ((yy == 3) & (pp >= 4)) | ((yy >= 4) & (pp == 3))
        types = pd.Series([f"L{a}->L{b}" for a, b in zip(yy[e_], pp[e_])]).value_counts().head(3).to_dict()
        print(f"  {NAME[c]}-{TR[tr]:3s}: 행 {len(ii):5d} 오류율 {e_.mean():.3f} 전체오류의 {e_.sum() / err.sum():.1%} | L2<->L3 {l23.sum() / max(e_.sum(), 1):.0%} | L4~L6 {late.sum() / max(e_.sum(), 1):.0%} | {types}")
    print("L2->L3 전환 시각 오차(예측이 처음 L3 가 된 시각 - 실제 L3 시작, 초): 음수=일찍 예측")
    for (c, tr), g in fr.groupby(["cls", "tr"]):
        errs = []
        for e in g.experiment_id.unique():
            ix = idx[e]; yy = y[ix]; pp = p[ix]
            if (yy >= 3).any() and (pp >= 3).any(): errs.append(t[ix][pp >= 3].min() - t[ix][yy >= 3].min())
        errs = np.array(errs)
        print(f"  {NAME[c]}-{TR[tr]:3s}: 실험 {len(errs):2d} | 중앙값 {np.median(errs):+6.0f}s | 절대 중앙 {np.median(np.abs(errs)):5.0f}s | |오차|>60s {np.mean(np.abs(errs) > 60):.0%} | 일찍(<-30s) {np.mean(errs < -30):.0%} 늦게(>30s) {np.mean(errs > 30):.0%}")
