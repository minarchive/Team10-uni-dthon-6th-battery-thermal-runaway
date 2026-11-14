"""손실 설계 비교(LightGBM, final_lgb 와 같은 피처·설정, OOF 5-fold).
V0 기준(다중분류)  V1 경계 가중(경계 +-W초 프레임 가중 1+g)  V2 시간 허용 소프트 라벨(경계 근처 프레임을 인접 단계로 복제, 가중 a)  V3 순서형 누적 분해(5개 이진 모델 y>=k)
각 변형을 과충전 L3 규칙 결합(thermal-atmp, theta 는 학습 fold)과 함께도 평가. 가중·소프트 라벨은 학습 라벨에서만 계산(검증 fold 라벨 미사용)."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); GAS = list(F.GAS)
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; oc_l = (fr.tr_method.values == 2)
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=14, verbose=-1, random_state=42)
d = raw.copy(); s = d[GAS].sum(axis=1)
for g in GAS: d[g] = 0.0
d["meas_co2"] = s
base = F.build_features(d); eid = d.experiment_id.values; t_all = d.relative_time_s.values.astype(float); idxs = pd.Series(np.arange(len(d))).groupby(eid).indices
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
X = pd.concat([base, pd.DataFrame(X1, index=base.index)], axis=1).iloc[lab_i].reset_index(drop=True)
tha = (d.thermal_t - d.atmp_t).values.astype(float); tha_cm = np.zeros(len(d))
for e, ix in idxs.items(): ix = ix[np.argsort(t_all[ix], kind="stable")]; tha_cm[ix] = np.maximum.accumulate(tha[ix])
tha_l = tha_cm[lab_i]
# 경계까지 거리(초)와 인접 단계(라벨 기반, 학습 행에서만 사용)
tl = fr.relative_time_s.values.astype(float); el = fr.experiment_id.values; dist = np.full(len(y), np.inf); adj = y.copy()
for e, ix in pd.Series(np.arange(len(y))).groupby(el).indices.items():
    ix = ix[np.argsort(tl[ix], kind="stable")]; yy = y[ix]; tt = tl[ix]; ch = np.where(np.diff(yy) != 0)[0] + 1
    for c in ch:
        for j in range(len(ix)):
            dd = abs(tt[j] - tt[c] + (tt[c] - tt[c - 1]) / 2) if False else min(abs(tt[j] - tt[c]), abs(tt[j] - tt[c - 1]))
            if dd < dist[ix[j]]:
                dist[ix[j]] = dd; adj[ix[j]] = yy[c] if j < c else yy[c - 1]
def fit_pred(Xtr, ytr, w, Xva):
    m = lgb.LGBMClassifier(**P0).fit(Xtr, ytr, sample_weight=w); Z = np.zeros((len(Xva), 6)); Z[:, np.array(m.classes_) - 1] = m.predict_proba(Xva); return Z
def theta_of(mask):
    v = tha_l[mask]; yy = (y[mask] == 3).astype(int); grid = np.unique(np.percentile(v, np.linspace(2, 98, 33))); b, bt = -1, grid[0]
    for g in grid:
        f = f1_score(yy, (v >= g).astype(int), zero_division=0)
        if f > b: b, bt = f, g
    return bt
def with_override(O):
    Oh = O.copy()
    for k in range(5):
        va = fold == k; th_ = theta_of((~va) & oc_l & np.isin(y, [2, 3])); rows = np.where(va & oc_l)[0]; P = Oh[rows]; l3 = tha_l[rows] >= th_; p23 = P[:, 1] + P[:, 2]; sel = (P.argmax(1) >= 1) & (P.argmax(1) <= 2)
        P[sel, 1] = np.where(l3[sel], 0.0, p23[sel]); P[sel, 2] = np.where(l3[sel], p23[sel], 0.0); Oh[rows] = P
    return Oh
fb = None
def rep(nm, O):
    global fb
    for tag, P in [("", O), (" +과충전규칙", with_override(O))]:
        p = F.postprocess(fr, P, 0.7); fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)]); fv = f1(y, p)
        if fb is None: fb = (fv, fs)
        print(f"{nm + tag:44s} OOF {fv:.4f} ({fv - fb[0]:+.4f}) 이긴 fold {int((fs > fb[1]).sum())}/5 | 과충전 {f1(y[oc_l], p[oc_l]):.4f} 가열 {f1(y[~oc_l], p[~oc_l]):.4f} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
def run(wfun=None, soft=None):
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; tr = np.where(~va)[0]; w = np.ones(len(tr)) if wfun is None else wfun(tr)
        Xtr, ytr = X.iloc[tr], y[tr]
        if soft is not None:
            D, a_ = soft; sel = tr[dist[tr] <= D]; aw = a_ * (1 - dist[sel] / (D + 5)); Xtr = pd.concat([Xtr, X.iloc[sel]]); ytr = np.concatenate([ytr, adj[sel]]); w = np.concatenate([w * 1.0, aw]); w[np.where(np.isin(tr, sel))[0]] *= 1.0
            sw = np.ones(len(tr)); pos = {r: i for i, r in enumerate(tr)}
            for r_, a__ in zip(sel, aw): sw[pos[r_]] -= a__
            w[:len(tr)] = sw
        O[va] = fit_pred(Xtr, ytr, w, X.iloc[va])
    return O
rep("V0 기준", run())
for W, g in [(60, 1.0), (120, 2.0)]:
    rep(f"V1 경계 가중 (±{W}s, +{g})", run(wfun=lambda tr, W=W, g=g: 1.0 + g * (dist[tr] <= W)))
for D, a_ in [(10, 0.3), (20, 0.3)]:
    rep(f"V2 소프트 라벨 (±{D}s, a={a_})", run(soft=(D, a_)))
# V3 순서형 누적 분해
O = np.zeros((len(y), 6))
for k in range(5):
    va = fold == k; tr = ~va; Pk = []
    for c in range(2, 7):
        m = lgb.LGBMClassifier(**P0).fit(X[tr], (y[tr] >= c).astype(int)); Pk.append(m.predict_proba(X[va])[:, 1])
    Pk = np.minimum.accumulate(np.stack(Pk, 1), axis=1); full = np.concatenate([np.ones((va.sum(), 1)), Pk, np.zeros((va.sum(), 1))], 1)
    O[va] = np.clip(full[:, :-1] - full[:, 1:], 1e-6, None) / np.clip(full[:, :-1] - full[:, 1:], 1e-6, None).sum(1, keepdims=True)
rep("V3 순서형 누적 분해", O)
