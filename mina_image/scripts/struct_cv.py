"""구조 질문 검증(OOF 5-fold, fast 설정).
A 형태별 모델: shared(전체 1개) / per_class(bat_class 별 3개) / per_cell(class x tr_method 5개) / warm(전체 모델에서 이어서 클래스별 추가 학습)  [X1 피처 캐시]
B 시간 해상도: 모든 실험을 stride 2(10초)/3(15초)로 줄여 피처 재계산·학습·평가, 같은 행에서 5초 기준 OOF(sensor_fast_oof)와 비교
C 가스 피처 제거: 가스 10종 전부 제거 / 가스합 하나만 남김."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7)); BASE0 = list(F.BASE_SIGNALS); DER0 = list(F.DERIVED)
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab = (raw.label.values >= 1) & (raw.fold.values >= 0); lab_i = np.where(lab)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; cls = fr.bat_class.values; trm = fr.tr_method.values
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=16, verbose=-1, random_state=42)
def full_proba(m, X): 
    pr = m.predict_proba(X); P = np.zeros((len(X), 6)); P[:, np.array(m.classes_) - 1] = pr; return P
def report(nm, O, base_f=None, base_fs=None, rows=None, frame=None, yy=None, fo=None):
    frame = fr if frame is None else frame; yy = y if yy is None else yy; fo = fold if fo is None else fo
    p = F.postprocess(frame, O, 0.7); fs = np.array([f1(yy[fo == k], p[fo == k]) for k in range(5)]); fv = f1(yy, p)
    extra = "" if base_f is None else f" ({fv - base_f:+.4f}) 이긴 fold {int((fs > base_fs).sum())}/5"
    print(f"{nm:30s} OOF {fv:.4f}{extra}", flush=True); return fv, fs
# ---------------- A
X1 = pd.read_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet")
print("== A. 배터리 형태별 처리 (X1 피처)")
def fit_oof(groups):
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k
        for g in np.unique(groups):
            tr = (~va) & (groups == g); te = va & (groups == g)
            if te.any(): O[te] = full_proba(lgb.LGBMClassifier(**P0).fit(X1[tr], y[tr]), X1[te])
    return O
sh = fit_oof(np.zeros(len(y), int)); bf, bfs = report("shared(기준)", sh)
report("per_class (3개 모델)", fit_oof(cls), bf, bfs)
report("per_cell (class x tr, 5개)", fit_oof(cls * 10 + trm), bf, bfs)
O = np.zeros((len(y), 6))
for k in range(5):
    va = fold == k; g = lgb.LGBMClassifier(**P0).fit(X1[~va], y[~va])
    for c in (1, 2, 3):
        tr = (~va) & (cls == c); te = va & (cls == c)
        m = lgb.LGBMClassifier(**{**P0, "n_estimators": 80, "learning_rate": 0.03}).fit(X1[tr], y[tr], init_model=g.booster_)
        O[te] = full_proba(m, X1[te])
report("warm (전체 -> 클래스별 이어서)", O, bf, bfs)
# ---------------- B, C
base5 = np.load(ROOT / "outputs" / "preds" / "sensor_fast_oof.npy")
def feats(df, drop_gas=False, gas_sum_only=False):
    F.BASE_SIGNALS = [s for s in BASE0 if not (drop_gas and s in GAS) and not (gas_sum_only and s in GAS)]
    F.DERIVED = [s for s in DER0 if not (drop_gas and s == "gas_sum")]
    F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    return F.build_features(df)
print("\n== B. 시간 해상도 (같은 행에서 5초 기준과 비교)")
for stride in (2, 3):
    keep = np.zeros(len(raw), bool)
    for e, ix in pd.Series(np.arange(len(raw))).groupby(raw.experiment_id.values).indices.items():
        ix = ix[np.argsort(raw.relative_time_s.values[ix], kind="stable")]; keep[ix[::stride]] = True
    sub = raw[keep].reset_index(drop=True); Xs = feats(sub)
    labs = (sub.label.values >= 1) & (sub.fold.values >= 0); ys = sub.label.values[labs].astype(int); fs_ = sub.fold.values[labs]; frs = sub[labs].reset_index(drop=True)
    O = np.zeros((len(ys), 6)); Xl = Xs[labs].reset_index(drop=True)
    for k in range(5):
        va = fs_ == k; O[va] = full_proba(lgb.LGBMClassifier(**P0).fit(Xl[~va], ys[~va]), Xl[va])
    # 같은 행(keep 의 라벨 행)에서 5초 기준 OOF 의 확률을 가져와 비교
    sid = frs.sample_id.values; pos = pd.Series(np.arange(len(fr)), index=fr.sample_id.values).loc[sid].values
    Ob = base5[pos]; fb = report(f"5초 기준(해당 행만, stride {stride})", Ob, frame=frs, yy=ys, fo=fs_)
    report(f"stride {stride} ({5 * stride}초) 재학습", O, fb[0], fb[1], frame=frs, yy=ys, fo=fs_)
print("\n== C. 가스 피처 제거 (5초 기준 sensor_fast_oof 와 비교)")
fb = report("가스 포함(기준)", base5)
for nm, kw in [("가스 10종 전부 제거", dict(drop_gas=True)), ("가스합만 유지", dict(gas_sum_only=True))]:
    Xg = feats(raw, **kw).iloc[lab_i].reset_index(drop=True); O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; O[va] = full_proba(lgb.LGBMClassifier(**P0).fit(Xg[~va], y[~va]), Xg[va])
    report(nm + f" (feat {Xg.shape[1]})", O, fb[0], fb[1])
