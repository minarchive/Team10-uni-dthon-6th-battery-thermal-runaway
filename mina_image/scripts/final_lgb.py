"""최종 LightGBM 후보: 가스 정리(가스합만) + thermal_t 임계(X1) + 과충전 L3 규칙 결합(thermal_t-atmp_t 누적최댓값 >= theta, theta 는 학습 실험의 과충전 L2/L3 프레임에서 F1 최적) + 팀 후처리.
1) OOF 5-fold (theta 는 학습 fold 안에서)  2) 지표: 풀링 OOF, Test 구성 재가중, 가상 Test  3) Train+Val 전체로 학습 -> Test CSV (이미지 블렌드 없음; 이미지를 쓰는 후보는 신경망 쪽)
--n_jobs 로 스레드 제한(다른 작업과 CPU 공유 시)."""
import argparse, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
ap = argparse.ArgumentParser(); ap.add_argument("--n_jobs", type=int, default=8); ap.add_argument("--name", default="final_lgb"); ap.add_argument("--full", action="store_true", help="500 트리 lr 0.03(최종 설정). 기본은 250 트리 lr 0.06(OOF 비교와 같은 설정)")
a = ap.parse_args()
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); GAS = list(F.GAS)
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]; te_i = np.where(raw.label.values == -1)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; oc_l = (fr.tr_method.values == 2)
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
P0 = dict(n_estimators=500 if a.full else 250, learning_rate=0.03 if a.full else 0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=a.n_jobs, verbose=-1, random_state=42)
d = raw.copy(); s = d[GAS].sum(axis=1)
for g in GAS: d[g] = 0.0
d["meas_co2"] = s                                                             # 가스 정리: 가스합만 유지
base = F.build_features(d)
eid = d.experiment_id.values; t_all = d.relative_time_s.values.astype(float); idxs = pd.Series(np.arange(len(d))).groupby(eid).indices
def cross(x, levels):
    out = {}
    for lv in levels:
        o = np.full(len(d), np.nan)
        for e, ix in idxs.items():
            ix = ix[np.argsort(t_all[ix], kind="stable")]; cm = np.maximum.accumulate(x[ix]); m = cm >= lv; tf = t_all[ix][np.argmax(m)] if m.any() else np.inf; o[ix] = np.where(t_all[ix] >= tf, t_all[ix] - tf, np.nan)
        out[lv] = o
    return out
th = d.thermal_t.values.astype(float); thr = th - pd.Series(th).groupby(eid).transform("first").values
X1 = {f"cx_thermal_{k}": v for k, v in cross(th, [35, 40, 45, 50, 55, 60, 65, 70, 80, 100, 150, 200]).items()}; X1.update({f"cx_thermal_rel_{k}": v for k, v in cross(thr, [2, 5, 10, 15, 20, 30, 40]).items()})
X = pd.concat([base, pd.DataFrame(X1, index=base.index)], axis=1)
tha = (d.thermal_t - d.atmp_t).values.astype(float); tha_cm = np.zeros(len(d))
for e, ix in idxs.items(): ix = ix[np.argsort(t_all[ix], kind="stable")]; tha_cm[ix] = np.maximum.accumulate(tha[ix])
oc_all = (d.tr_method.values == 2)
def pick_theta(rows_mask):                                                   # 과충전 L2/L3 프레임에서 F1 최적 임계
    v = tha_cm[rows_mask]; grid = np.unique(np.nanpercentile(v, np.linspace(2, 98, 33))); yy = (d.label.values[rows_mask] == 3).astype(int); b, bt = -1, grid[0]
    for g in grid:
        f = f1_score(yy, (v >= g).astype(int), zero_division=0)
        if f > b: b, bt = f, g
    return bt
def override(P, rows, theta):
    P = P.copy(); l3 = tha_cm[rows] >= theta; p23 = P[:, 1] + P[:, 2]; sel = (P.argmax(1) >= 1) & (P.argmax(1) <= 2)
    P[sel, 1] = np.where(l3[sel], 0.0, p23[sel]); P[sel, 2] = np.where(l3[sel], p23[sel], 0.0); return P
Xl = X.iloc[lab_i].reset_index(drop=True); O = np.zeros((len(y), 6)); Oh = np.zeros((len(y), 6))
for k in range(5):
    va = fold == k; m = lgb.LGBMClassifier(**P0).fit(Xl[~va], y[~va]); Z = np.zeros((va.sum(), 6)); Z[:, np.array(m.classes_) - 1] = m.predict_proba(Xl[va]); O[va] = Z
    trm_full = np.zeros(len(d), bool); trm_full[lab_i[(~va) & oc_l & np.isin(y, [2, 3])]] = True; theta = pick_theta(trm_full)
    Zh = Z.copy(); vi = np.where(va)[0]; rr = np.where(oc_l[vi])[0]           # fold 안 과충전 행(fold 내부 위치)
    if len(rr): Zh[rr] = override(Z[rr], lab_i[vi[rr]], theta)
    Oh[va] = Zh
np.save(ROOT / "outputs/preds" / f"{a.name}_oof.npy", Oh.astype(np.float32)); np.save(ROOT / "outputs/preds" / f"{a.name}_oof_nooverride.npy", O.astype(np.float32))
for nm, P in [("모델만", O), ("모델 + 과충전 규칙 결합", Oh)]:
    p = F.postprocess(fr, P, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    print(f"{nm:24s} OOF {f1(y, p):.4f} | fold {np.round(fs, 3).tolist()} | 과충전 {f1(y[oc_l], p[oc_l]):.4f} 가열 {f1(y[~oc_l], p[~oc_l]):.4f} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
# 최종: Train+Val 로 학습 -> Test
mf = lgb.LGBMClassifier(**P0).fit(Xl, y); Xt = X.iloc[te_i].reset_index(drop=True); Pt = np.zeros((len(te_i), 6)); Pt[:, np.array(mf.classes_) - 1] = mf.predict_proba(Xt)
all_oc = np.zeros(len(d), bool); all_oc[lab_i[oc_l & np.isin(y, [2, 3])]] = True; theta_f = pick_theta(all_oc)
rows = np.where(oc_all[te_i])[0]; Pt[rows] = override(Pt[rows], te_i[rows], theta_f)
ft = raw.iloc[te_i][["sample_id", "experiment_id", "relative_time_s"]].reset_index(drop=True); lab_te = F.postprocess(ft, Pt, 0.7)
order = pd.read_csv(DATA / "sample_submission.csv", dtype={"sample_id": "string"})["sample_id"].astype(str).tolist()
sr = pd.Series(lab_te.astype(int), index=ft.sample_id.values); sub = pd.DataFrame({"sample_id": order, "label": sr.loc[order].values})
assert len(sub) == 3451 and sub.label.isin(L).all() and sub.sample_id.is_unique
out = ROOT / "outputs/submissions" / f"{a.name}.csv"; sub.to_csv(out, index=False); np.save(ROOT / "outputs/preds" / f"{a.name}_test.npy", Pt.astype(np.float32))
ref = pd.read_csv(ROOT / "outputs/submissions/ens_lgb_xgb.csv", dtype={"sample_id": "string"}).set_index("sample_id").label
print(f"theta={theta_f:.2f} | 저장 {out.name} | 분포 {sub.label.value_counts(normalize=True).sort_index().round(3).to_dict()} | ens_lgb_xgb 와 일치율 {(ref.loc[sub.sample_id.values].values == sub.label.values).mean():.4f} | 과충전 test 행 {len(rows)}")
