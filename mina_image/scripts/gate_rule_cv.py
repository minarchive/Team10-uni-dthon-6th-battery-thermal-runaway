"""규칙 상태 게이팅 검증: 상태(T1 45°C 도달 / 전압 급락)로 허용 단계를 제한했을 때 규칙 위반률과 센서 LightGBM OOF 의 F1 변화.
상태 0: T1 누적최댓값<45 -> {L1}.  상태 1: 도달 후·급락 전 -> {L2,L3}.  상태 2: 급락 후 -> {L4,L5,L6}.  급락 = 전압비 누적최솟값 <= thr 또는 한 칸 하락 >= d (과거만 사용)."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=["sample_id", "experiment_id", "relative_time_s", "meas_t1", "meas_v"]) for s in ["train", "validation"]])
raw["sample_id"] = raw.sample_id.astype(str); raw = raw.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index(); raw["experiment_id"] = raw.experiment_id.astype(str)
eid = raw.experiment_id.values; t = raw.relative_time_s.values.astype(float); t1 = raw.meas_t1.values.astype(float); v = raw.meas_v.values.astype(float)
so = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy")
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
def states(thr, step):
    s = np.zeros(len(y), int)
    for e, ix in pd.Series(np.arange(len(y))).groupby(eid).indices.items():
        ix = ix[np.argsort(t[ix], kind="stable")]; cm = np.maximum.accumulate(t1[ix]); vr = v[ix] / max(v[ix][0], 1e-6)
        cv = np.minimum.accumulate(vr); dv = -np.r_[0, np.diff(vr)]; drop = np.maximum.accumulate(((cv <= thr) if thr else np.zeros(len(ix), bool)) | ((dv >= step) if step else np.zeros(len(ix), bool)))
        st = np.where(cm >= 45, 1, 0); st = np.where((cm >= 45) & drop, 2, st); s[ix] = st
    return s
ALLOW = {0: [1], 1: [2, 3], 2: [4, 5, 6]}
pb = F.postprocess(fr, so, 0.7); base = f1(y, pb); print(f"기준 OOF {base:.4f} | 단계별 {np.round(f1_score(y, pb, labels=L, average=None), 3).tolist()}\n")
for nm, thr, step in [("v/v0<=0.8", 0.8, 0), ("v/v0<=0.9", 0.9, 0), ("v/v0<=0.95", 0.95, 0), ("한 칸 하락>=0.1", 0, 0.1), ("(<=0.9 또는 한 칸>=0.1)", 0.9, 0.1)]:
    s = states(thr, step)
    viol = np.array([y[i] not in ALLOW[s[i]] for i in range(len(y))])
    P = so.copy(); mask = np.zeros_like(P)
    for st, al in ALLOW.items(): mask[np.ix_(np.where(s == st)[0], np.array(al) - 1)] = 1
    Pg = P * mask; sm = Pg.sum(1, keepdims=True); Pg = np.where(sm > 0, Pg / np.where(sm > 0, sm, 1), P)   # 허용 클래스 확률이 전부 0 이면 원본 유지
    pg = F.postprocess(fr, Pg, 0.7); fs = np.array([f1(y[fold == k], pg[fold == k]) for k in range(5)]); fb = np.array([f1(y[fold == k], pb[fold == k]) for k in range(5)])
    print(f"{nm:24s} 규칙 위반 프레임 {int(viol.sum()):5d} ({viol.mean():.2%}) [상태별 {[int(viol[s == k].sum()) for k in range(3)]}] | 게이팅 후 OOF {f1(y, pg):.4f} ({f1(y, pg) - base:+.4f}) 이긴 fold {int((fs > fb).sum())}/5 | 단계별 {np.round(f1_score(y, pg, labels=L, average=None), 3).tolist()}")
