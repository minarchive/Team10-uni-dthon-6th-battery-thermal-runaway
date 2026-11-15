"""팀 LightGBM(재현) 위의 CPU 실험 스위트: 모델 계열, 정규화, 분석 기반 변형, 배깅, 앙상블, 마지막에 압축 진단 + 증강.
모든 변형은 (1) 전체 길이 OOF (2) 균등 500행 압축 OOF(Test 와 같은 cap_rows, 피처 재계산) 를 같이 보고한다.
실행: python -u src/team_suite.py [--only B0,A1] [--skip_aug]
"""
import argparse
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import team_pipeline as T

F, log = T.F, T.log
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--skip_aug", action="store_true")
a = ap.parse_args()
ONLY = set(x for x in a.only.split(",") if x)

df = T.load_table()
Lb = df[df.split != "test"].reset_index(drop=True)
lab_i = np.where(df.split != "test")[0]
y = Lb.label.to_numpy(); fold = Lb.fold.to_numpy(); oc = (Lb.tr_method.values == 2); expid = Lb.experiment_id.to_numpy()
cap = F.cap_rows(Lb); cap_idx = cap.index.to_numpy(); cap = cap.reset_index(drop=True)
y_c = cap.label.to_numpy(); oc_c = (cap.tr_method.values == 2)
fold_of_exp = Lb.groupby("experiment_id").fold.first().to_dict(); fold_c = cap.experiment_id.map(fold_of_exp).to_numpy()
log(f"전체 {len(Lb)}행 -> 압축 {len(cap)}행 (500행 초과 실험 {int((Lb.groupby('experiment_id').size() > 500).sum())}개), 과충전 행 {int(oc.sum())} -> {int(oc_c.sum())}")

VIEWS = {}


def view(gas):
    if gas not in VIEWS:
        X = T.get_features(df, "gas_" + gas, gas).iloc[lab_i].reset_index(drop=True)
        pc = T.CACHE / f"team_Xcap_{gas}.parquet"
        if pc.exists():
            Xc = pd.read_parquet(pc)
        else:
            t0 = time.time(); Xc = T.build_X(cap.assign(sample_id=cap.sample_id), gas); Xc.to_parquet(pc); log(f"압축 피처 {gas}: {Xc.shape} {time.time() - t0:.0f}s")
        VIEWS[gas] = (X, Xc)
    return VIEWS[gas]


# ---------------------------------------------------------------- 과충전 덮어쓰기 (임계 / 로지스틱)
def ovr_feats(fr):
    t = fr.relative_time_s.values.astype(float); eid = fr.experiment_id.values
    th, t1, t2, am = (fr[c].values.astype(float) for c in ("thermal_t", "meas_t1", "meas_t2", "atmp_t"))
    a1, a2, a3 = np.zeros(len(fr)), np.zeros(len(fr)), np.zeros(len(fr))
    for _, ix in pd.Series(np.arange(len(fr))).groupby(eid).indices.items():
        ix = ix[np.argsort(t[ix], kind="stable")]
        a1[ix] = np.maximum.accumulate((th - am)[ix]); a2[ix] = np.maximum.accumulate((t1 - t1[ix][0])[ix]); a3[ix] = np.minimum.accumulate((t2 - t1)[ix])
    return np.c_[a1, a2, a3]


OV_F, OV_Fc = ovr_feats(Lb), ovr_feats(cap)


def apply_override(Pf, Pc, mode):
    if mode == "none":
        return Pf, Pc
    Pf, Pc = Pf.copy(), Pc.copy()
    for k in range(5):
        m = (fold != k) & oc & np.isin(y, [2, 3])
        if mode == "thr":
            th = T.pick_theta(OV_F[:, 0], y, m)
            rows = np.where((fold == k) & oc)[0]; rc = np.where((fold_c == k) & oc_c)[0]
            if len(rows): Pf[rows] = T.override(Pf[rows], OV_F[rows, 0], th)
            if len(rc): Pc[rc] = T.override(Pc[rc], OV_Fc[rc, 0], th)
        else:  # lr3
            mdl = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500)).fit(OV_F[m], (y[m] == 3).astype(int))
            for P, OF, fo, oo in ((Pf, OV_F, fold, oc), (Pc, OV_Fc, fold_c, oc_c)):
                rows = np.where((fo == k) & oo)[0]
                if len(rows):
                    l3 = mdl.predict(OF[rows]).astype(bool); p23 = P[rows, 1] + P[rows, 2]; sel = (P[rows].argmax(1) >= 1) & (P[rows].argmax(1) <= 2)
                    r = rows[sel]; P[r, 1] = np.where(l3[sel], 0.0, p23[sel]); P[r, 2] = np.where(l3[sel], p23[sel], 0.0)
    return Pf, Pc


# ---------------------------------------------------------------- 학습 함수들
def fit_lgb(**kw):
    def f(Xtr, ytr, w):
        m = T.make_lgb(**kw).fit(Xtr, ytr, sample_weight=w); return lambda X: T.proba6(m, X)
    return f


def fit_xgb(**kw):
    import xgboost as xgb
    def f(Xtr, ytr, w):
        cnt = np.bincount(ytr, minlength=7)[1:]; cw = (len(ytr) / (6 * np.maximum(cnt, 1)))[ytr - 1]; sw = cw if w is None else cw * w
        p = dict(n_estimators=250, learning_rate=0.06, max_depth=6, subsample=0.8, colsample_bytree=0.3, min_child_weight=3, reg_lambda=1.0, tree_method="hist", max_bin=127, n_jobs=-1, random_state=42, verbosity=0)
        p.update(kw); m = xgb.XGBClassifier(**p).fit(Xtr, ytr - 1, sample_weight=sw); return lambda X: m.predict_proba(X)
    return f


def fit_et(**kw):
    from sklearn.ensemble import ExtraTreesClassifier
    def f(Xtr, ytr, w):
        p = dict(n_estimators=150, max_features=0.1, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=42); p.update(kw)
        try:
            m = ExtraTreesClassifier(**p).fit(Xtr, ytr, sample_weight=w); fill = None
        except Exception:
            m = ExtraTreesClassifier(**p).fit(Xtr.fillna(-999), ytr, sample_weight=w); fill = -999
        def pr(X):
            Z = np.zeros((len(X), 6)); Z[:, m.classes_.astype(int) - 1] = m.predict_proba(X if fill is None else X.fillna(fill)); return Z
        return pr
    return f


def fit_bag(n_bags=6, by_exp=True, **kw):
    def f(Xtr, ytr, w, ctx):
        eids = ctx['eids']; preds = []
        rng = np.random.default_rng(0); ue = np.unique(eids)
        for b in range(n_bags):
            if by_exp:
                pick = rng.choice(ue, len(ue), replace=True); cnt = pd.Series(pick).value_counts()
                ww = pd.Series(eids).map(cnt).fillna(0).to_numpy(float)
            else:
                ww = np.ones(len(ytr))
            m = T.make_lgb(n_estimators=150, learning_rate=0.08, random_state=42 + b, **kw).fit(Xtr[ww > 0], ytr[ww > 0], sample_weight=ww[ww > 0]); preds.append(m)
        return lambda X: np.mean([T.proba6(m, X) for m in preds], axis=0)
    f.needs_ctx = True
    return f


def fit_seeds(seeds=(42, 43, 44), **kw):
    def f(Xtr, ytr, w):
        ms = [T.make_lgb(random_state=s, **kw).fit(Xtr, ytr, sample_weight=w) for s in seeds]; return lambda X: np.mean([T.proba6(m, X) for m in ms], axis=0)
    return f


def boundary_w(tr_idx, width=60, g=1.0):
    w = np.ones(len(tr_idx)); sub = Lb.iloc[tr_idx]; t = sub.relative_time_s.to_numpy(float); lb = sub.label.to_numpy(); e = sub.experiment_id.to_numpy()
    for _, ix in pd.Series(np.arange(len(sub))).groupby(e).indices.items():
        ix = ix[np.argsort(t[ix], kind="stable")]; ch = np.where(np.diff(lb[ix]) != 0)[0] + 1
        for c in ch:
            near = np.abs(t[ix] - t[ix][c]) <= width; w[ix[near]] += g
    return w



# ---------------------------------------------------------------- 방식별 모델 / 압력 절대값 제거 / Viterbi
P_ABS = re.compile(r"^meas_p(_cummax|_cummin|_mean\d+|_max\d+|_min\d+)?$")


class ByMethod:
    needs_meth = True

    def __init__(self, models):
        self.models = models

    def __call__(self, X, meth):
        P = np.zeros((len(X), 6))
        for m_, (mdl, cols) in self.models.items():
            sel = meth == m_
            if sel.any():
                P[sel] = T.proba6(mdl, X[sel][cols])
        return P


def fit_bymethod(drop_oc=(), drop_heat=(), **kw):
    def f(Xtr, ytr, w, ctx):
        meth = Lb.tr_method.values[ctx["tr_idx"]]; models = {}
        for m_ in (1, 2):
            sel = meth == m_; cols = [c for c in Xtr.columns if c not in (drop_oc if m_ == 2 else drop_heat)]
            mdl = T.make_lgb(**kw).fit(Xtr[sel][cols], ytr[sel], sample_weight=None if w is None else w[sel]); models[m_] = (mdl, cols)
        return ByMethod(models)
    f.needs_ctx = True
    return f


def viterbi(df, P):
    """접두 Viterbi(팀 sota_0.8812 와 같은 정의): 각 시점까지의 확률로 '단계가 내려가지 않는' 경로의 MAP 마지막 상태. 하이퍼파라미터 없음."""
    pred = np.zeros(len(df), int); t = df.relative_time_s.values
    for _, idx in df.reset_index(drop=True).groupby("experiment_id").indices.items():
        idx = idx[np.argsort(t[idx], kind="stable")]; Lg = np.log(np.clip(P[idx], 1e-6, 1)); best = None
        for j, i in enumerate(idx):
            best = Lg[j] if best is None else np.maximum.accumulate(best) + Lg[j]; pred[i] = best.argmax() + 1
    return pred


def decode(frame, P, dec):
    return F.postprocess(frame, P, 0.7) if dec == "conf" else viterbi(frame, P)


STRESS_D = (10, 25)   # Test 각형 과충전의 시작 압력(12.9, 31.5)은 학습 최대 4.25 의 3~7배 -> 과충전 압력에 +10/+25 를 더한 분포 이동


def stress_view(gas, cols, d):
    pth = T.CACHE / f"team_Xs_{gas}_{d}.parquet"
    if pth.exists():
        Xs = pd.read_parquet(pth)
    else:
        fr = Lb.copy(); fr.loc[fr.tr_method == 2, "meas_p"] += d; t0 = time.time(); Xs = T.build_X(fr, gas); Xs.to_parquet(pth); log(f"스트레스 피처 {gas} +{d}: {time.time() - t0:.0f}s")
    return Xs if cols is None else Xs[cols]


# ---------------------------------------------------------------- CV 실행 + 보고
STORE, ROWS, BASE = {}, [], {}


def cv(fit_fn, gas="sum", cols=None, wfun=None, extra=None, stress=False):
    X, Xc = view(gas)
    if cols is not None:
        X, Xc = X[cols], Xc[cols]
    XS = {d: stress_view(gas, cols, d) for d in STRESS_D} if stress else {}
    Pf, Pc = np.zeros((len(y), 6)), np.zeros((len(cap), 6)); Ps = {d: np.zeros((len(y), 6)) for d in XS}
    meth_l, meth_c = Lb.tr_method.values, cap.tr_method.values
    for k in range(5):
        va, vac = fold == k, fold_c == k; tr = np.where(~va)[0]
        Xtr, ytr = X.iloc[tr], y[tr]; w = wfun(tr) if wfun else None
        if extra is not None:
            Xe, ye = extra(k, X.columns); Xtr = pd.concat([Xtr, Xe], ignore_index=True); ytr = np.concatenate([ytr, ye])
            w = None if w is None else np.concatenate([w, np.ones(len(ye))])
        pred = fit_fn(Xtr, ytr, w, dict(tr_idx=tr, eids=expid[tr])) if getattr(fit_fn, "needs_ctx", False) else fit_fn(Xtr, ytr, w)
        call = (lambda Z, m: pred(Z, m)) if getattr(pred, "needs_meth", False) else (lambda Z, m: pred(Z))
        Pf[va] = call(X[va], meth_l[va]); Pc[vac] = call(Xc[vac], meth_c[vac])
        for d_, Xs in XS.items():
            Ps[d_][va] = call(Xs[va], meth_l[va])
    return Pf, Pc, Ps


def evaluate(name, Pf, Pc, mode="thr", save=True, decs=("conf",), Ps=None):
    if save:
        STORE[name] = (Pf, Pc)
    Of, Oc = apply_override(Pf, Pc, mode)
    pr3 = oc & (Lb.bat_class.values == 3)
    for dec in decs:
        pf, pc = decode(Lb, Of, dec), decode(cap, Oc, dec); nm = name if dec == "conf" else name + "|vit"
        r = dict(name=nm, override=mode, oof_full=T.f1m(y, pf), oof_cap=T.f1m(y_c, pc), oc_full=T.f1m(y[oc], pf[oc]), oc_cap=T.f1m(y_c[oc_c], pc[oc_c]),
                 heat_full=T.f1m(y[~oc], pf[~oc]), heat_cap=T.f1m(y_c[~oc_c], pc[~oc_c]), pr_full=T.f1m(y[pr3], pf[pr3]))
        if Ps:
            for d_, P in Ps.items():
                Os, _ = apply_override(P, np.zeros((len(cap), 6)), mode); ps = decode(Lb, Os, dec)
                r[f"oc_s{d_}"] = T.f1m(y[oc], ps[oc]); r[f"pr_s{d_}"] = T.f1m(y[pr3], ps[pr3])
        for tag, p, yy, ee in (("full", pf, y, expid), ("cap", pc, y_c, cap.experiment_id.values)):
            if "B0" in BASE:
                bp = BASE["B0"][tag]; a = pd.Series(p == yy).groupby(ee).mean(); b = pd.Series(bp == yy).groupby(ee).mean(); dl = (a - b).to_numpy()
                rng = np.random.default_rng(0); bs = [dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(2000)]
                r[f"d_{tag}"] = dl.mean(); r[f"ci_lo_{tag}"] = np.percentile(bs, 2.5); r[f"ci_hi_{tag}"] = np.percentile(bs, 97.5)
        if nm == "B0":
            BASE["B0"] = dict(full=pf, cap=pc)
        ROWS.append(r); pd.DataFrame(ROWS).to_csv(T.OUT / "suite_results.csv", index=False)
        s = (f"{nm:36s}[{mode}] 전체 {r['oof_full']:.4f} 압축 {r['oof_cap']:.4f} | 과충전 전체 {r['oc_full']:.4f} 압축 {r['oc_cap']:.4f} (각형과충전 {r['pr_full']:.3f}) | 가열 {r['heat_full']:.4f}/{r['heat_cap']:.4f}")
        if Ps:
            s += " | 압력이동 " + " ".join(f"+{d_}: 과충전 {r[f'oc_s{d_}']:.4f} 각형과충전 {r[f'pr_s{d_}']:.3f}" for d_ in Ps)
        if "d_full" in r:
            s += f" | B0 대비 Δ전체 {r['d_full']:+.4f}[{r['ci_lo_full']:+.4f},{r['ci_hi_full']:+.4f}] Δ압축 {r['d_cap']:+.4f}[{r['ci_lo_cap']:+.4f},{r['ci_hi_cap']:+.4f}]"
        log(s)


def run(name, fit_fn, gas="sum", cols=None, wfun=None, extra=None, mode="thr", stress=False, decs=("conf",)):
    if ONLY and name not in ONLY and name != "B0":
        return
    t0 = time.time(); ff, fc = T.OUT / f"suite_{name}_full.npy", T.OUT / f"suite_{name}_cap.npy"; fs = {d: T.OUT / f"suite_{name}_s{d}.npy" for d in STRESS_D}
    if ff.exists() and fc.exists() and (not stress or all(p.exists() for p in fs.values())):
        Pf, Pc = np.load(ff), np.load(fc); Ps = {d: np.load(p) for d, p in fs.items()} if stress else {}
        log(f"  ({name}: 저장된 결과 사용)")
    else:
        Pf, Pc, Ps = cv(fit_fn, gas, cols, wfun, extra, stress)
        np.save(ff, Pf); np.save(fc, Pc)
        for d_, P in Ps.items():
            np.save(fs[d_], P)
    evaluate(name, Pf, Pc, mode, decs=decs, Ps=Ps or None); log(f"  ({name}: {time.time() - t0:.0f}s)")


# ================================================================= 우선순위 큐
Xs0, _ = view("sum"); ALLC = list(Xs0.columns)
DROP_P = [c for c in ALLC if P_ABS.match(c)]
log(f"압력 절대수준 피처 {len(DROP_P)}개: {DROP_P[:6]}...")

# [우선순위 1] 분포 이동 대응 + 스트레스 테스트 (+ 디코딩 비교: conf vs 접두 Viterbi)
run("B0", fit_lgb(), stress=True, decs=("conf", "viterbi"))
evaluate("B0_model_only", *STORE["B0"], mode="none", save=False)
Pf, Pc = STORE["B0"]; evaluate("C4_B0+override_lr3", Pf, Pc, mode="lr3", save=False)
run("S1_dropP_all", fit_lgb(), cols=[c for c in ALLC if c not in DROP_P], stress=True, decs=("conf", "viterbi"))
run("S2_bymethod_dropP_oc", fit_bymethod(drop_oc=DROP_P), stress=True, decs=("conf", "viterbi"))
run("S3_bymethod_dropP_gasnone", fit_bymethod(drop_oc=DROP_P), gas="none", stress=True, decs=("conf", "viterbi"))

# [우선순위 2] 증강 (과충전 압축 진단: 과충전이 압축에서 -0.012)
if not a.skip_aug:
    log("=== 진단: B0 의 전체 길이 vs 균등 500행 압축 (방식별) ===")
    r0 = [r for r in ROWS if r["name"] == "B0"][0]
    log(f"과충전 F1 전체 {r0['oc_full']:.4f} -> 압축 {r0['oc_cap']:.4f} (Δ{r0['oc_cap'] - r0['oc_full']:+.4f}) | 가열 {r0['heat_full']:.4f} -> {r0['heat_cap']:.4f} (Δ{r0['heat_cap'] - r0['heat_full']:+.4f})")

    def aug_copies(kind, n_copies=2, seed=0):
        """학습용 증강 사본: 간격 증강(stride 1~3, 자료집 명시 간격만) 또는 위상 이동 압축(500행 초과 실험만). 같은 experiment_id 유지(같은 fold)."""
        rng = np.random.default_rng(seed); outs = []
        for c in range(n_copies):
            keep = []
            for e, ix in Lb.groupby("experiment_id", sort=False).indices.items():
                ix = ix[np.argsort(Lb.relative_time_s.values[ix], kind="stable")]
                if kind == "stride":
                    p, sel = 0, []
                    while p < len(ix):
                        sel.append(ix[p]); p += rng.choice([1, 2, 3], p=[0.6, 0.25, 0.15])
                    keep.extend(sel)
                elif len(ix) > 500:
                    step = len(ix) / 500; pos = np.unique(np.minimum(((np.arange(500) + rng.uniform(0, 1)) * step).astype(int), len(ix) - 1)); keep.extend(ix[pos])
            if keep:
                outs.append(Lb.iloc[np.sort(np.array(keep))].copy())
        return outs

    def make_extra(kind, n_copies=2):
        parts = []
        for c, d_ in enumerate(aug_copies(kind, n_copies)):
            d2 = d_.copy(); d2["sample_id"] = d2.sample_id + f"_a{kind}{c}"
            Xa = T.build_X(d2.reset_index(drop=True), "sum"); parts.append((Xa, d2.label.to_numpy(), d2.experiment_id.map(fold_of_exp).to_numpy()))
        def extra(k, cols):
            return pd.concat([p[0][cols][p[2] != k] for p in parts], ignore_index=True), np.concatenate([p[1][p[2] != k] for p in parts])
        return extra
    t0 = time.time(); ex_stride = make_extra("stride"); log(f"간격 증강 사본 피처 {time.time() - t0:.0f}s")
    run("F1_aug_stride2", fit_lgb(), extra=ex_stride)
    t0 = time.time(); ex_phase = make_extra("phase"); log(f"위상 이동 압축 사본 피처 {time.time() - t0:.0f}s")
    run("F2_aug_phasecap2", fit_lgb(), extra=ex_phase)
    def ex_both(k, cols):
        a1, b1 = ex_stride(k, cols); a2, b2 = ex_phase(k, cols); return pd.concat([a1, a2], ignore_index=True), np.concatenate([b1, b2])
    run("F3_aug_both", fit_lgb(), extra=ex_both)

# [우선순위 3] 가스·윈도·가중
run("C1_gas_none", fit_lgb(), gas="none")
run("C2_prune_windows", fit_lgb(), cols=[c for c in ALLC if not re.search(r"_(mean|std|max|min|dev|slope|delta)(120|300)$", c)])
run("C3_boundary_weight", fit_lgb(), wfun=lambda tr: boundary_w(tr))

# [우선순위 4] 배깅 / 시드 / 모델 계열 (저장된 결과는 불러옴)
run("A1_xgb", fit_xgb()); run("A2_extratrees", fit_et())
run("B1_shallow_reg", fit_lgb(num_leaves=15, min_child_samples=150, colsample_bytree=0.2, reg_lambda=10.0, extra_trees=True))
run("B2_trees500_lr03", fit_lgb(n_estimators=500, learning_rate=0.03)); run("B3_min_child200", fit_lgb(min_child_samples=200)); run("B4_colsample015", fit_lgb(colsample_bytree=0.15))
run("D1_exp_bagging6", fit_bag(6, True)); run("D2_row_bagging6", fit_bag(6, False)); run("D3_seed_avg3", fit_seeds())


def blend(name, parts, w=None):
    if any(p not in STORE for p in parts):
        return
    w = w or [1 / len(parts)] * len(parts)
    evaluate(name, sum(wi * STORE[p][0] for wi, p in zip(w, parts)), sum(wi * STORE[p][1] for wi, p in zip(w, parts)), "thr", save=False)
blend("E1_LGB+XGB", ["B0", "A1_xgb"]); blend("E2_LGB+XGB+ET", ["B0", "A1_xgb", "A2_extratrees"]); blend("E3_LGB+ET", ["B0", "A2_extratrees"])
blend("E4_LGB+bag6", ["B0", "D1_exp_bagging6"]); blend("E5_LGB+XGB+bag6", ["B0", "A1_xgb", "D1_exp_bagging6"])
blend("E6_all", ["B0", "A1_xgb", "A2_extratrees", "D1_exp_bagging6", "B1_shallow_reg"]); blend("E7_gasnone+LGB", ["B0", "C1_gas_none"])
Of, Oc = apply_override(*STORE["B0"], "thr")
for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
    pf, pc = F.postprocess(Lb, Of, tau), F.postprocess(cap, Oc, tau)
    log(f"τ={tau}: 전체 {T.f1m(y, pf):.4f} 압축 {T.f1m(y_c, pc):.4f} | 과충전 전체 {T.f1m(y[oc], pf[oc]):.4f} 압축 {T.f1m(y_c[oc_c], pc[oc_c]):.4f}")

res = pd.DataFrame(ROWS); res.to_csv(T.OUT / "suite_results.csv", index=False)
log("=== 요약 (전체 OOF 순) ===")
print(res.sort_values("oof_full", ascending=False)[["name", "oof_full", "oof_cap", "oc_full", "oc_cap", "pr_full", "d_full", "ci_lo_full", "ci_hi_full", "d_cap"]].round(4).to_string(index=False), flush=True)
st = [c for c in res.columns if c.startswith(("oc_s", "pr_s"))]
if st:
    print("\n=== 압력 분포 이동 스트레스 (과충전 압력 +10/+25) ===")
    print(res[res[st].notna().any(axis=1)][["name", "oc_full", "pr_full"] + st].round(4).to_string(index=False), flush=True)
