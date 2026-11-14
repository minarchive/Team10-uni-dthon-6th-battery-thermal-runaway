"""최종 LightGBM 후보: 가스 정리(가스합만) + thermal_t 임계(X1) + 과충전 L3 규칙 결합(thermal_t-atmp_t 누적최댓값 >= theta, theta 는 학습 실험의 과충전 L2/L3 프레임에서 F1 최적) + 팀 후처리.
1) OOF 5-fold (theta 는 학습 fold 안에서)  2) 지표: 풀링 OOF, Test 구성 재가중, 가상 Test  3) Train+Val 전체로 학습 -> Test CSV (이미지 블렌드 없음; 이미지를 쓰는 후보는 신경망 쪽)
--n_jobs 로 스레드 제한(다른 작업과 CPU 공유 시)."""
import argparse, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
ap = argparse.ArgumentParser(); ap.add_argument("--n_jobs", type=int, default=8); ap.add_argument("--name", default="final_lgb"); ap.add_argument("--extra", default="", help="rate=45℃ 도달 후 상승속도 계열 인과 피처 추가"); ap.add_argument("--aug_oc", default="", help="과충전 학습 실험을 시간축 배속(예 0.8,1.25)으로 재표본한 복사본을 학습에만 추가(평가는 원본만)"); ap.add_argument("--aug_mode", default="warp", choices=["warp", "cap500", "offset"]); ap.add_argument("--cap_rows", type=int, default=500, help="cap500 모드의 재표본 행 수"); ap.add_argument("--w_soc", type=float, default=1.0, help="SOC 100% 실험 프레임의 학습 가중(1=동일)"); ap.add_argument("--test_dist", type=int, default=0, help="1=학습 가중을 Test 그룹 행 구성에 맞춤(scripts/test_dist.py)"); ap.add_argument("--xgb", type=int, default=0, help="1=같은 피처·가중으로 XGBoost 도 학습해 확률을 0.5:0.5 평균"); ap.add_argument("--td_lo", type=float, default=0.2); ap.add_argument("--td_hi", type=float, default=5.0); ap.add_argument("--td_pow", type=float, default=1.0, help="그룹 배율 = (Test 비중/학습 비중)^pow, [td_lo, td_hi] 클립"); ap.add_argument("--img", default="", help="이미지 모델 확률(예: ft_dinov2_b)을 인과 피처(현재·누적최대·최근 평균)로 LightGBM 입력에 추가"); ap.add_argument("--oc_rule", default="thr", choices=["thr", "logit", "speed"], help="과충전 L3 덮어쓰기 규칙: thr=thermal-atmp 누적최대 단일 임계(기존), logit=인과 3개 피처 로지스틱, speed=45℃ 도달 후 60초 상승속도(인과)로 임계를 빠름/느림 2구간으로"); ap.add_argument("--eval_aug_only", type=int, default=0, help="1=증강 복사본을 학습에 쓰지 않고 평가에만 사용(압축 형식 강건성 측정)"); ap.add_argument("--drop_p", type=int, default=0, help="1=압력 절대 수준 피처(meas_p, 누적최대·최소, 롤링 평균·최대·최소) 제거 -> 상승분·기울기만 사용(Test 과충전 시작 압력이 학습 범위를 크게 벗어나는 것에 대한 강건성)"); ap.add_argument("--by_method", type=int, default=0, help="1=방식(가열/과충전)별로 따로 학습·예측"); ap.add_argument("--no_override", type=int, default=0, help="1=과충전 L3 덮어쓰기 규칙을 쓰지 않음(Test 비교용)"); ap.add_argument("--aug_all", type=int, default=0, help="1=가열 실험도 증강 대상"); ap.add_argument("--full", action="store_true", help="500 트리 lr 0.03(최종 설정). 기본은 250 트리 lr 0.06(OOF 비교와 같은 설정)")
a = ap.parse_args()
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); GAS = list(F.GAS)
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
if a.aug_oc:      # 학습 전용 복사본(원본과 같은 fold, 평가는 원본만). 모드: warp=시간축 배속 / cap500=500행 균등 간격 재표본(Test 형식 모사) / offset=thermal_t·atmp_t 상수 오프셋 지터
    _stat = {"sample_id", "experiment_id", "relative_time_s", "fold", "label"}; _num = [c for c in raw.columns if c not in _stat and pd.api.types.is_float_dtype(raw[c])]; _new = []; _rng = np.random.default_rng(0)
    _sel = (raw.label >= 1) & (raw.fold >= 0) & ((raw.tr_method == 2) | bool(a.aug_all))
    for _e, _g in raw[_sel].groupby("experiment_id"):
        _g = _g.sort_values("relative_time_s", kind="stable"); _t = _g.relative_time_s.values.astype(float)
        if a.aug_mode == "warp":
            _outs = []
            for _sp in [float(v) for v in a.aug_oc.split(",")]:
                _tau = np.arange(0, _t[-1] / _sp + 1e-9, 5.0) * _sp; _tau = _tau[_tau <= _t[-1]]; _h = _g.iloc[np.clip(np.searchsorted(_t, _tau, side="right") - 1, 0, len(_t) - 1)].copy()
                for _c in _num: _h[_c] = np.interp(_tau, _t, _g[_c].values.astype(float))
                _h["relative_time_s"] = np.arange(len(_tau)) * 5.0; _outs.append((f"aug{_sp}", _h))
        elif a.aug_mode == "cap500":
            if len(_g) <= a.cap_rows: continue
            _outs = [(f"aug{_k}", _g.iloc[np.unique(np.round(np.linspace(_k, len(_g) - 1, a.cap_rows)).astype(int))].copy()) for _k in range(int(a.aug_oc))]
        else:
            _outs = []
            for _k in range(int(a.aug_oc)):
                _h = _g.copy(); _h["thermal_t"] = _h.thermal_t + _rng.uniform(-3, 3); _h["atmp_t"] = _h.atmp_t + _rng.uniform(-3, 3); _outs.append((f"aug{_k}", _h))
        for _nm, _h in _outs:
            _h["experiment_id"] = f"{_e}_{_nm}"; _h["sample_id"] = [f"{_e}_{_nm}_{i}" for i in range(len(_h))]; _new.append(_h)
    raw = pd.concat([raw] + _new, ignore_index=True); print(f"증강 복사본 {len(_new)}개, 행 {sum(len(h) for h in _new)} (모드 {a.aug_mode})", flush=True)
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]; te_i = np.where(raw.label.values == -1)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); ev = ~fr.experiment_id.astype(str).str.contains("_aug").values; y = fr.label.astype(int).values; fold = fr.fold.values; oc_l = (fr.tr_method.values == 2)
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
if a.drop_p:
    _dp = [c for c in X.columns if c.startswith("meas_p") and any(k in c for k in ("mean", "max", "min")) or c == "meas_p"]; X = X.drop(columns=_dp); print("압력 절대 수준 피처 제거:", len(_dp), "개", flush=True)
if a.img:      # 이미지 모델 확률(라벨 행은 OOF, Test 는 fold 평균)을 인과 피처로: 현재 6개 + P(>=L3), P(>=L4) 의 누적최대·최근 6/12프레임 평균
    _N0 = len(tb); IMG = np.full((len(d), 6), np.nan); _lo = np.where((tb.label.values >= 1) & (tb.fold.values >= 0))[0]; _te0 = np.where(tb.label.values == -1)[0]
    IMG[_lo] = np.load(ROOT / "outputs/preds" / f"{a.img}_oof.npy"); IMG[_te0] = np.load(ROOT / "outputs/preds" / f"{a.img}_test.npy")
    XI = {f"img_p{k + 1}": IMG[:, k] for k in range(6)}; g3 = IMG[:, 2:].sum(1); g4 = IMG[:, 3:].sum(1)
    for nm, g in (("ge3", g3), ("ge4", g4)):
        cm, r6, r12 = np.full(len(d), np.nan), np.full(len(d), np.nan), np.full(len(d), np.nan)
        for e, ix in idxs.items():
            ix = ix[np.argsort(t_all[ix], kind="stable")]; sr = pd.Series(g[ix]); cm[ix] = sr.cummax().values; r6[ix] = sr.rolling(6, min_periods=1).mean().values; r12[ix] = sr.rolling(12, min_periods=1).mean().values
        XI[f"img_{nm}_cm"], XI[f"img_{nm}_r6"], XI[f"img_{nm}_r12"] = cm, r6, r12
    X = pd.concat([X, pd.DataFrame(XI, index=base.index)], axis=1)
if a.extra == "rate":     # 인과: T1 누적최대가 45℃ 이상이 된 '이후' 행에서만 값. 60/120초 시점 속도는 그 시점이 지난 행부터(그 전 NaN)
    XR = {n: np.full(len(d), np.nan) for n in ["since45", "t1_rate", "th_rate", "t1_r60", "t1_r120", "th_r60", "th_r120", "since_x_t1rate", "since_x_t1r60", "since_x_thr60", "since_x_thrate"]}
    t1v = d.meas_t1.values.astype(float); thv = d.thermal_t.values.astype(float)
    for e, ix in idxs.items():
        ix = ix[np.argsort(t_all[ix], kind="stable")]; tt = t_all[ix]; cm = np.maximum.accumulate(t1v[ix])
        if not (cm >= 45).any(): continue
        j = int(np.argmax(cm >= 45)); n = len(ix); pos = np.arange(n); aft = pos >= j; el = np.where(aft, tt - tt[j], np.nan)
        r = {}
        for nm, x in (("t1", t1v[ix]), ("th", thv[ix])):
            rate = np.where(pos > j, (x - x[j]) / np.maximum(tt - tt[j], 5.0), np.nan); r[nm + "_rate"] = rate
            for w in (60, 120):
                k_ = int(np.searchsorted(tt, tt[j] + w, side="left")); o = np.full(n, np.nan)
                if k_ < n: o[k_:] = (x[k_] - x[j]) / max(tt[k_] - tt[j], 5.0)
                r[f"{nm}_r{w}"] = o
        for nm in ("t1_rate", "th_rate", "t1_r60", "t1_r120", "th_r60", "th_r120"): XR[nm][ix] = r[nm]
        XR["since45"][ix] = el; XR["since_x_t1rate"][ix] = el * r["t1_rate"]; XR["since_x_t1r60"][ix] = el * r["t1_r60"]; XR["since_x_thr60"][ix] = el * r["th_r60"]; XR["since_x_thrate"][ix] = el * r["th_rate"]
    X = pd.concat([X, pd.DataFrame(XR, index=base.index)], axis=1)
tha = (d.thermal_t - d.atmp_t).values.astype(float); tha_cm = np.zeros(len(d))
for e, ix in idxs.items(): ix = ix[np.argsort(t_all[ix], kind="stable")]; tha_cm[ix] = np.maximum.accumulate(tha[ix])
oc_all = (d.tr_method.values == 2)
def pick_theta(rows_mask):                                                   # 과충전 L2/L3 프레임에서 F1 최적 임계
    v = tha_cm[rows_mask]; grid = np.unique(np.nanpercentile(v, np.linspace(2, 98, 33))); yy = (d.label.values[rows_mask] == 3).astype(int); b, bt = -1, grid[0]
    for g in grid:
        f = f1_score(yy, (v >= g).astype(int), zero_division=0)
        if f > b: b, bt = f, g
    return bt
t1r_cm = np.zeros(len(d)); t21_cm = np.zeros(len(d)); R60 = np.full(len(d), np.nan); _t1 = d.meas_t1.values.astype(float); _t2 = d.meas_t2.values.astype(float)
for e, ix in idxs.items():
    ix = ix[np.argsort(t_all[ix], kind="stable")]; t1r_cm[ix] = np.maximum.accumulate(_t1[ix] - _t1[ix][0]); t21_cm[ix] = np.minimum.accumulate(_t2[ix] - _t1[ix]); cmx = np.maximum.accumulate(_t1[ix])
    if (cmx >= 45).any():
        j = int(np.argmax(cmx >= 45)); k_ = int(np.searchsorted(t_all[ix], t_all[ix][j] + 60, side="left"))
        if k_ < len(ix): R60[ix[k_:]] = (_t1[ix][k_] - _t1[ix][j]) / max(t_all[ix][k_] - t_all[ix][j], 5.0)        # 도달 60초 뒤부터 값(그 전 NaN, 미래 값 미사용)
FS = np.column_stack([tha_cm, t1r_cm, t21_cm])
def make_rule(mask, th):
    if a.oc_rule == "thr": return lambda r: tha_cm[r] >= th
    if a.oc_rule == "logit":
        from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=300)).fit(FS[mask], (d.label.values[mask] == 3).astype(int)); return lambda r: clf.predict(FS[r]) == 1
    kn = mask & ~np.isnan(R60); med = np.median(R60[kn]) if kn.sum() > 20 else 0.0; thf = pick_theta(kn & (R60 >= med)) if (kn & (R60 >= med)).sum() > 20 else th; ths = pick_theta(kn & (R60 < med)) if (kn & (R60 < med)).sum() > 20 else th
    return lambda r: np.where(np.isnan(R60[r]), tha_cm[r] >= th, np.where(R60[r] >= med, tha_cm[r] >= thf, tha_cm[r] >= ths))
def override(P, rows, theta):
    P = P.copy(); l3 = theta(rows); p23 = P[:, 1] + P[:, 2]; sel = (P.argmax(1) >= 1) & (P.argmax(1) <= 2)
    P[sel, 1] = np.where(l3[sel], 0.0, p23[sel]); P[sel, 2] = np.where(l3[sel], p23[sel], 0.0); return P
def mk_pred(Xa, ya, wa, Xb, tra=None, trb=None):
    if a.by_method and tra is not None:
        out = np.zeros((len(Xb), 6))
        for m_ in (1, 2):
            ia, ib = tra == m_, trb == m_
            if ib.any(): out[ib] = mk_pred1(Xa[ia], ya[ia], wa[ia], Xb[ib])
        return out
    return mk_pred1(Xa, ya, wa, Xb)
def mk_pred1(Xa, ya, wa, Xb):
    m = lgb.LGBMClassifier(**P0).fit(Xa, ya, sample_weight=wa); Z = np.zeros((len(Xb), 6)); Z[:, np.array(m.classes_) - 1] = m.predict_proba(Xb)
    if a.xgb:
        import xgboost as xgb; from sklearn.utils.class_weight import compute_sample_weight
        mx = xgb.XGBClassifier(n_estimators=P0["n_estimators"], learning_rate=P0["learning_rate"], max_depth=6, subsample=0.8, colsample_bytree=0.3, reg_lambda=1.0, tree_method="hist", n_jobs=a.n_jobs, verbosity=0, random_state=42).fit(Xa, ya - 1, sample_weight=compute_sample_weight("balanced", ya) * wa)
        Zx = np.zeros((len(Xb), 6)); Zx[:, mx.classes_] = mx.predict_proba(Xb); Z = 0.5 * Z + 0.5 * Zx
    return Z
trn = ev if a.eval_aug_only else np.ones(len(fr), bool)
SW = np.where(fr.soc_status.values == 2, a.w_soc, 1.0)
if a.test_dist:
    import test_dist as TD; _m = TD.group_mult(a.td_lo, a.td_hi, a.td_pow); print("그룹 배율(Test/학습 행 비중):", {k: round(v, 2) for k, v in _m.items()}, flush=True); SW = SW * TD.row_weights(fr.bat_class.values, fr.tr_method.values, fr.experiment_id.astype(str).values, _m)
Xl = X.iloc[lab_i].reset_index(drop=True); O = np.zeros((len(y), 6)); Oh = np.zeros((len(y), 6))
for k in range(5):
    va = fold == k; Z = mk_pred(Xl[(~va) & trn], y[(~va) & trn], SW[(~va) & trn], Xl[va], fr.tr_method.values[(~va) & trn], fr.tr_method.values[va]); O[va] = Z
    trm_full = np.zeros(len(d), bool); trm_full[lab_i[(~va) & trn & oc_l & np.isin(y, [2, 3])]] = True; theta = pick_theta(trm_full); rule_k = make_rule(trm_full, theta)
    Zh = Z.copy(); vi = np.where(va)[0]; rr = np.where(oc_l[vi])[0]           # fold 안 과충전 행(fold 내부 위치)
    if len(rr) and not a.no_override: Zh[rr] = override(Z[rr], lab_i[vi[rr]], rule_k)
    Oh[va] = Zh
np.save(ROOT / "outputs/preds" / f"{a.name}_oof.npy", Oh[ev].astype(np.float32)); np.save(ROOT / "outputs/preds" / f"{a.name}_oof_nooverride.npy", O[ev].astype(np.float32))
for nm, P in [("모델만", O), ("모델 + 과충전 규칙 결합", Oh)]:
    p = F.postprocess(fr, P, 0.7); fs = np.array([f1(y[ev & (fold == k)], p[ev & (fold == k)]) for k in range(5)])
    print(f"{nm:24s} OOF {f1(y[ev], p[ev]):.4f} | fold {np.round(fs, 3).tolist()} | 과충전 {f1(y[ev & oc_l], p[ev & oc_l]):.4f} 가열 {f1(y[ev & ~oc_l], p[ev & ~oc_l]):.4f} | 단계별 {np.round(f1_score(y[ev], p[ev], labels=L, average=None), 3).tolist()}", flush=True)
    if (~ev).any(): print(f"   └ 압축(500행 균등 간격) 복사본 행 OOF F1 {f1(y[~ev], p[~ev]):.4f} (복사본 {int((~ev).sum())}행; 원본 같은 fold 의 검증 쪽 복사본)", flush=True)
# 최종: Train+Val 로 학습 -> Test
Xt = X.iloc[te_i].reset_index(drop=True); Pt = mk_pred(Xl[trn], y[trn], SW[trn], Xt, fr.tr_method.values[trn], raw.tr_method.values[te_i])
all_oc = np.zeros(len(d), bool); all_oc[lab_i[trn & oc_l & np.isin(y, [2, 3])]] = True; theta_f = pick_theta(all_oc)
rows = np.where(oc_all[te_i])[0]
if not a.no_override: Pt[rows] = override(Pt[rows], te_i[rows], make_rule(all_oc, theta_f))
ft = raw.iloc[te_i][["sample_id", "experiment_id", "relative_time_s"]].reset_index(drop=True); lab_te = F.postprocess(ft, Pt, 0.7)
order = pd.read_csv(DATA / "sample_submission.csv", dtype={"sample_id": "string"})["sample_id"].astype(str).tolist()
sr = pd.Series(lab_te.astype(int), index=ft.sample_id.values); sub = pd.DataFrame({"sample_id": order, "label": sr.loc[order].values})
assert len(sub) == 3451 and sub.label.isin(L).all() and sub.sample_id.is_unique
out = ROOT / "outputs/submissions" / f"{a.name}.csv"; sub.to_csv(out, index=False); np.save(ROOT / "outputs/preds" / f"{a.name}_test.npy", Pt.astype(np.float32))
ref = pd.read_csv(ROOT / "outputs/submissions/ens_lgb_xgb.csv", dtype={"sample_id": "string"}).set_index("sample_id").label
print(f"theta={theta_f:.2f} | 저장 {out.name} | 분포 {sub.label.value_counts(normalize=True).sort_index().round(3).to_dict()} | ens_lgb_xgb 와 일치율 {(ref.loc[sub.sample_id.values].values == sub.label.values).mean():.4f} | 과충전 test 행 {len(rows)}")
