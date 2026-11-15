"""리뷰 논문 기반 물리 피처 시험 (CPU): (1) Arrhenius 계열 ln(dT/dt) vs 1/T, (2) 가열 방식 히터 대비 자체 발열.
B0(final_lgb 재현) 피처 + 새 피처 -> 같은 5-fold LGBM. 전체 길이 / 균등 500행 압축 둘 다, B0 대비 실험별 대응 부트스트랩.
모든 새 피처는 같은 실험의 현재·과거만 사용 (마지막에 앞 60% 절단 인과성 검사).
실행: python -u src/team_physics.py
"""
import time

import numpy as np
import pandas as pd

import team_pipeline as T

F, log = T.F, T.log
df = T.load_table()
Lb = df[df.split != "test"].reset_index(drop=True)
lab_i = np.where(df.split != "test")[0]
y = Lb.label.to_numpy(); fold = Lb.fold.to_numpy(); oc = Lb.tr_method.values == 2; expid = Lb.experiment_id.to_numpy()
cap = F.cap_rows(Lb).reset_index(drop=True)
y_c = cap.label.to_numpy(); oc_c = cap.tr_method.values == 2
fe = Lb.groupby("experiment_id").fold.first().to_dict(); fold_c = cap.experiment_id.map(fe).to_numpy()


def lagval(t, v, w):
    """시각 t-w 이하의 마지막 관측값 (없으면 NaN). t 는 정렬됨."""
    j = np.searchsorted(t, t - w, side="right") - 1
    out = np.where(j >= 0, v[np.clip(j, 0, None)], np.nan)
    return out


def phys(fr):
    n = len(fr); t_all = fr.relative_time_s.values.astype(float); eid = fr.experiment_id.values
    th = fr.thermal_t.values.astype(float); ind = fr.indu_t.values.astype(float)
    cols = {}
    names = []

    def put(name, arr):
        if name not in cols:
            cols[name] = np.full(n, np.nan); names.append(name)
        return cols[name]
    for _, ix in pd.Series(np.arange(n)).groupby(eid).indices.items():
        ix = ix[np.argsort(t_all[ix], kind="stable")]; t = t_all[ix]; T_ = th[ix]; Tk = T_ + 273.15; H = ind[ix]
        r = {}
        for w in (30, 60, 120):
            r[w] = (T_ - lagval(t, T_, w)) / w
            put(f"ph_rate{w}", None)[ix] = r[w]
        rr = np.clip(r[60], 1e-3, None)
        lr = np.log(rr); put("ph_lnrate60", None)[ix] = lr
        lr_prev = lagval(t, lr, 60); inv = 1.0 / Tk; inv_prev = lagval(t, inv, 60)
        dinv = inv - inv_prev
        ea = np.where(np.abs(dinv) > 1e-6, -(lr - lr_prev) / dinv, np.nan)
        ea = np.clip(ea, -2e4, 2e4); put("ph_ea_app", None)[ix] = ea
        put("ph_rate60_cummax", None)[ix] = np.fmax.accumulate(np.nan_to_num(r[60], nan=-9))
        put("ph_ea_cummax", None)[ix] = np.fmax.accumulate(np.nan_to_num(ea, nan=-2e4))
        put("ph_rate_accel", None)[ix] = (r[30] - lagval(t, r[30], 60))
        put("ph_rate_rel", None)[ix] = r[60] / (np.abs(T_ - T_[0]) + 1.0)
        # 히터 대비 (가열 방식만 값 있음)
        if np.isfinite(H).any():
            hr = (H - lagval(t, H, 60)) / 60; put("ph_heater_rate60", None)[ix] = hr
            d = r[60] - hr; put("ph_selfheat_diff", None)[ix] = d
            put("ph_selfheat_cummax", None)[ix] = np.fmax.accumulate(np.nan_to_num(d, nan=-9))
            put("ph_heater_gap", None)[ix] = H - T_
            put("ph_heater_gap_cummin", None)[ix] = np.fmin.accumulate(np.nan_to_num(H - T_, nan=1e9))
            cm = np.fmax.accumulate(np.nan_to_num(d, nan=-9))
            for lv in (0.0, 0.02, 0.05):
                m = cm > lv; tf = t[np.argmax(m)] if m.any() else np.inf
                put(f"ph_since_selfheat_{lv}", None)[ix] = np.where(t >= tf, t - tf, np.nan)
    return pd.DataFrame({k: cols[k] for k in names}, index=np.arange(n))


t0 = time.time()
Pl = phys(Lb); Pc_ = phys(cap); log(f"물리 피처 {Pl.shape[1]}개 {time.time() - t0:.0f}s: {list(Pl.columns)}")

# 인과성 검사: 앞 60% 만 가지고 만든 피처 == 전체로 만든 뒤 앞 60% 값
bad = []
sub = Lb.groupby("experiment_id", group_keys=False).apply(lambda g: g.sort_values("relative_time_s").iloc[: int(len(g) * 0.6)]).reset_index(drop=True)
ps = phys(sub); key = pd.Series(np.arange(len(Lb)), index=Lb.sample_id.values)
full_part = Pl.iloc[key.loc[sub.sample_id.values].values].reset_index(drop=True)
for c in Pl.columns:
    a, b = ps[c].to_numpy(), full_part[c].to_numpy(); ok = np.isclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-6)
    if not ok.all():
        bad.append(c)
log("인과성 검사:", "통과" if not bad else f"실패 {bad}")

X = T.get_features(df, "gas_sum", "sum").iloc[lab_i].reset_index(drop=True)
Xc = pd.read_parquet(T.CACHE / "team_Xcap_sum.parquet")
XA, XAc = pd.concat([X, Pl], axis=1), pd.concat([Xc, Pc_], axis=1)


def cvrun(Xf, Xcf, nj):
    Pf, Pc = np.zeros((len(y), 6)), np.zeros((len(cap), 6))
    for k in range(5):
        va, vac = fold == k, fold_c == k
        m = T.make_lgb(n_jobs=nj).fit(Xf[~va], y[~va])
        Pf[va] = T.proba6(m, Xf[va]); Pc[vac] = T.proba6(m, Xcf[vac]); log(f"  fold {k} 완료")
    return Pf, Pc


def ovr(fr):
    return T.tha_cummax(fr)


THf, THc = ovr(Lb), ovr(cap)


def post(Pf, Pc):
    Pf, Pc = Pf.copy(), Pc.copy()
    for k in range(5):
        th = T.pick_theta(THf, y, (fold != k) & oc & np.isin(y, [2, 3]))
        rows = np.where((fold == k) & oc)[0]; rc = np.where((fold_c == k) & oc_c)[0]
        if len(rows): Pf[rows] = T.override(Pf[rows], THf[rows], th)
        if len(rc): Pc[rc] = T.override(Pc[rc], THc[rc], th)
    return F.postprocess(Lb, Pf, 0.7), F.postprocess(cap, Pc, 0.7)


nj = 8
Bf, Bc = np.load(T.OUT / "suite_B0_full.npy"), np.load(T.OUT / "suite_B0_cap.npy")
bpf, bpc = post(Bf, Bc)
res = {}
for name, (Xf, Xcf) in {"P_all(+새 피처 전부)": (XA, XAc),
                        "P_arrh(+Arrhenius 계열만)": (pd.concat([X, Pl[[c for c in Pl.columns if "heater" not in c and "selfheat" not in c]]], axis=1), pd.concat([Xc, Pc_[[c for c in Pc_.columns if "heater" not in c and "selfheat" not in c]]], axis=1)),
                        "P_heat(+히터 대비만)": (pd.concat([X, Pl[[c for c in Pl.columns if "heater" in c or "selfheat" in c]]], axis=1), pd.concat([Xc, Pc_[[c for c in Pc_.columns if "heater" in c or "selfheat" in c]]], axis=1))}.items():
    t0 = time.time(); Pf, Pc = cvrun(Xf, Xcf, nj); pf, pc = post(Pf, Pc)
    np.save(T.OUT / f"phys_{name[:6]}_full.npy", Pf); np.save(T.OUT / f"phys_{name[:6]}_cap.npy", Pc)
    s = f"{name:28s} 전체 {T.f1m(y, pf):.4f} 압축 {T.f1m(y_c, pc):.4f} | 과충전 {T.f1m(y[oc], pf[oc]):.4f}/{T.f1m(y_c[oc_c], pc[oc_c]):.4f} 가열 {T.f1m(y[~oc], pf[~oc]):.4f}/{T.f1m(y_c[~oc_c], pc[~oc_c]):.4f}"
    for tag, p, bp, yy, ee in (("전체", pf, bpf, y, expid), ("압축", pc, bpc, y_c, cap.experiment_id.values)):
        a = pd.Series(p == yy).groupby(ee).mean(); b = pd.Series(bp == yy).groupby(ee).mean(); dl = (a - b).to_numpy()
        rng = np.random.default_rng(0); bs = [dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(2000)]
        s += f" | B0대비 {tag} Δ{dl.mean():+.4f}[{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] 개선{(dl > 0).sum()}/악화{(dl < 0).sum()}"
    log(s + f" ({time.time() - t0:.0f}s)")
log(f"기준 B0: 전체 {T.f1m(y, bpf):.4f} 압축 {T.f1m(y_c, bpc):.4f} | 과충전 {T.f1m(y[oc], bpf[oc]):.4f}/{T.f1m(y_c[oc_c], bpc[oc_c]):.4f} 가열 {T.f1m(y[~oc], bpf[~oc]):.4f}/{T.f1m(y_c[~oc_c], bpc[~oc_c]):.4f}")
log("완료")
