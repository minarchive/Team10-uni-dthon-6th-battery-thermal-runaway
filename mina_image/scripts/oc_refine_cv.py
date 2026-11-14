"""과충전 L3 규칙 정교화(모델 OOF 위, 학습 fold 에서만 임계 결정): 
 A 현재: 과충전 전체 하나의 theta (thermal-atmp 누적최댓값)
 B 형태별 theta (파우치/각형)   C 용량별 theta(bat_cc)   D 신호 다중: thermal-atmp 와 t1 상승량 둘 다 임계(AND/OR)  E 신호 교체: thermal_t 절대 누적최댓값"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=["sample_id", "experiment_id", "relative_time_s", "meas_t1", "thermal_t", "atmp_t", "bat_class", "bat_cc", "tr_method"]) for s in ["train", "validation"]])
raw["sample_id"] = raw.sample_id.astype(str); raw = raw.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index(); raw["experiment_id"] = raw.experiment_id.astype(str)
eid = raw.experiment_id.values; t = raw.relative_time_s.values.astype(float); idx = pd.Series(np.arange(len(y))).groupby(eid).indices
cls = raw.bat_class.values.astype(int); cc = raw.bat_cc.values.astype(float); oc = raw.tr_method.values == 2
def cumm(x):
    o = np.zeros(len(y))
    for e, ix in idx.items(): ix = ix[np.argsort(t[ix], kind="stable")]; o[ix] = np.maximum.accumulate(x[ix])
    return o
SIG = {"tha": cumm((raw.thermal_t - raw.atmp_t).values.astype(float)), "th": cumm(raw.thermal_t.values.astype(float)), "t1rel": cumm((raw.meas_t1 - raw.groupby("experiment_id").meas_t1.transform("first")).values.astype(float))}
Pn = np.load(ROOT / "outputs/preds/final_lgb_oof_nooverride.npy"); f1 = lambda p: f1_score(y, p, labels=L, average="macro", zero_division=0)
pb = F.postprocess(fr, np.load(ROOT / "outputs/preds/final_lgb_oof.npy"), 0.7); base = f1(pb); fb = np.array([f1_score(y[fold == k], pb[fold == k], labels=L, average="macro") for k in range(5)])
def best_th(v, yy):
    grid = np.unique(np.percentile(v, np.linspace(2, 98, 33))); b, th = -1, grid[0]
    for g in grid:
        f = f1_score(yy, (v >= g).astype(int), zero_division=0)
        if f > b: b, th = f, g
    return th
def override(P, rows, l3):
    p23 = P[rows, 1] + P[rows, 2]; sel = (P[rows].argmax(1) >= 1) & (P[rows].argmax(1) <= 2); r = rows[sel]; P[r, 1] = np.where(l3[sel], 0.0, p23[sel]); P[r, 2] = np.where(l3[sel], p23[sel], 0.0)
def run(nm, group_fn, sigs, mode="and"):
    P = Pn.copy()
    for k in range(5):
        va = fold == k; trm0 = (~va) & oc & np.isin(y, [2, 3])
        for g in np.unique(group_fn[oc]):
            gm = group_fn == g; trm = trm0 & gm; rows = np.where(va & oc & gm)[0]
            if len(rows) == 0 or trm.sum() < 50 or len(np.unique(y[trm])) < 2: continue
            ths = [best_th(SIG[s][trm], (y[trm] == 3).astype(int)) for s in sigs]
            conds = [SIG[s][rows] >= th for s, th in zip(sigs, ths)]; l3 = np.logical_and.reduce(conds) if mode == "and" else np.logical_or.reduce(conds); override(P, rows, l3)
    p = F.postprocess(fr, P, 0.7); fs = np.array([f1_score(y[fold == k], p[fold == k], labels=L, average="macro") for k in range(5)]); fv = f1(p)
    print(f"{nm:34s} OOF {fv:.4f} ({fv - base:+.4f}) 이긴 fold {int((fs > fb).sum())}/5 | 과충전 {f1_score(y[oc], p[oc], labels=L, average='macro'):.4f} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
print(f"기준(현재 규칙: 과충전 전체 theta, tha) {base:.4f} | 모델만(규칙 없음) {f1(F.postprocess(fr, Pn, 0.7)):.4f}")
one = np.zeros(len(y), int)
run("A 현재 재현 (tha, 과충전 전체)", one, ["tha"])
run("B 형태별 theta (tha)", cls, ["tha"])
run("C 용량별 theta (tha)", cc.astype(int), ["tha"])
run("E 신호 교체: thermal_t 절대", one, ["th"])
run("E2 신호 교체: t1 상승량", one, ["t1rel"])
run("D1 tha AND t1rel", one, ["tha", "t1rel"], "and")
run("D2 tha OR t1rel", one, ["tha", "t1rel"], "or")
