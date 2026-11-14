"""센서 확률 + 이미지(순서형) 확률을 작은 2단계 스태커로 결합한다. modality 별로 잘하는 구간을 스태커가 학습.

입력: outputs/preds/sensor_lgb_{oof,test}.npy (sensor_oof.py), ft_<img>_{oof,test}.npy (finetune.py)
스태커 피처(약 40개, 모두 인과적): log 센서확률 6 + log 이미지확률 6, 확신도·엔트로피, P(>=L3)·P(>=L4) 두 modality 와 차이,
   과거 60/180초 평균·누적최대. 805개 피처에 이미지 신호를 섞던 'light' 와 달리 저차원이라 과적합이 작다.
평가 3가지:
  A. 2단계 5-fold OOF(같은 folds.parquet): 센서 단독 vs 스태커
  B. 공식 프로토콜: 스태커는 Train 행(OOF 센서)으로만 학습 -> Validation. 센서 입력은 (B1) val0.9195 Train-only 모델 확률 / (B2) OOF 확률
     Validation 은 cap500(Test 처럼 500행 상한) 과 full 둘 다 보고
  C. --final: 모든 라벨 행으로 스태커 학습 -> Test(센서 fold 평균 + 이미지 fold 평균) -> 후처리 -> 제출 CSV
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import DATA, ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--img", default="ft_ord_dinov2_b", help="이미지 확률 태그(여러 개면 콤마, 확률 평균)")
ap.add_argument("--ts_dir", default=str(ROOT / "ts_model"))
ap.add_argument("--official_model", default="/home/ubuntu/nicole/val0.9195/model.joblib", help="Train-only 센서 모델(B1용). 없으면 B1 생략")
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--n_jobs", type=int, default=6)
ap.add_argument("--final", action="store_true")
ap.add_argument("--name", default="gate_stack")
a = ap.parse_args()
sys.path.insert(0, a.ts_dir)
import features as F
import lightgbm as lgb

T0 = time.time()
log = lambda *m: print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *m, flush=True)
L = list(range(1, 7))
P = ROOT / "outputs" / "preds"

df = table()
y_all, fold_all = df.label.values, df.fold.values
lab = (y_all >= 1) & (fold_all >= 0)
te = np.where(y_all == -1)[0]
lab_idx = np.where(lab)[0]
sens_oof = np.load(P / "sensor_lgb_oof.npy")
sens_te = np.load(P / "sensor_lgb_test.npy")
tags = a.img.split(",")
img_oof = np.mean([np.load(P / f"{t}_oof.npy") for t in tags], 0)
img_te = np.mean([np.load(P / f"{t}_test.npy") for t in tags], 0)
meta = df[["sample_id", "experiment_id", "relative_time_s", "split", "label", "fold"]].copy()
meta["experiment_id"], meta["sample_id"] = meta.experiment_id.astype(str), meta.sample_id.astype(str)
Ps_full = np.full((len(df), 6), np.nan, np.float32)
Pi_full = np.full((len(df), 6), np.nan, np.float32)
Ps_full[lab_idx], Pi_full[lab_idx] = sens_oof, img_oof
Ps_full[te], Pi_full[te] = sens_te, img_te
log(f"라벨 행 {lab.sum()}, test {len(te)}, 이미지={tags}")

lg = lambda p: np.log(np.clip(p, 1e-4, 1))


def sfeat(fr, Ps, Pi):
    """fr: experiment_id, relative_time_s 가진 프레임. Ps, Pi: (n,6). 행 순서 그대로 피처 DataFrame 반환."""
    n = len(fr)
    t = fr.relative_time_s.values.astype(float)
    e = fr.experiment_id.values
    order = np.lexsort((t, e))
    ent = lambda p: -(p * lg(p)).sum(1)
    base = {f"ls{k}": lg(Ps)[:, k - 1] for k in L}
    base.update({f"li{k}": lg(Pi)[:, k - 1] for k in L})
    sg3, sg4, ig3, ig4 = Ps[:, 2:].sum(1), Ps[:, 3:].sum(1), Pi[:, 2:].sum(1), Pi[:, 3:].sum(1)
    base.update(conf_s=Ps.max(1), conf_i=Pi.max(1), ent_s=ent(Ps), ent_i=ent(Pi), arg_s=Ps.argmax(1) + 1, arg_i=Pi.argmax(1) + 1,
                sg3=sg3, sg4=sg4, ig3=ig3, ig4=ig4, d3=ig3 - sg3, d4=ig4 - sg4,
                exp_s=Ps @ np.array(L), exp_i=Pi @ np.array(L))
    X = pd.DataFrame(base)
    tmp = pd.DataFrame({"e": e[order], "ts": pd.to_datetime(t[order], unit="s")})
    for nm in ["ig3", "ig4", "sg3", "sg4", "d3", "d4", "exp_s", "exp_i"]:
        tmp["v"] = X[nm].values[order]
        g = tmp.groupby("e", sort=False)
        for w in (60, 180):
            r = g.rolling(f"{w}s", on="ts", closed="right")["v"].mean().reset_index(drop=True).values
            col = np.empty(n)
            col[order] = r
            X[f"{nm}_m{w}"] = col
        cm = np.empty(n)
        cm[order] = g["v"].cummax().values
        X[f"{nm}_cmax"] = cm
    return X


def mk():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=2.0, class_weight="balanced", random_state=42, n_jobs=a.n_jobs, verbose=-1)


def f1(y, p):
    return f1_score(y, p, labels=L, average="macro", zero_division=0)


def post(fr, Pm):
    return F.postprocess(fr.reset_index(drop=True), Pm, a.tau)


fr_lab = meta.iloc[lab_idx].reset_index(drop=True)
y_lab = fr_lab.label.astype(int).values
X_lab = sfeat(fr_lab, sens_oof, img_oof)

# ------------------------------------------------------------ A. 2단계 OOF
log("A. 2단계 5-fold OOF")
oof2 = np.zeros((len(fr_lab), 6))
for f in range(5):
    va = (fr_lab.fold.values == f)
    m = mk().fit(X_lab[~va], y_lab[~va])
    oof2[va] = m.predict_proba(X_lab[va])
rowsA = []
for nm, Pm in [("센서 단독", sens_oof), ("이미지 단독", img_oof), ("스태커", oof2)]:
    fs = [f1(y_lab[fr_lab.fold.values == f], post(fr_lab[fr_lab.fold.values == f], Pm[fr_lab.fold.values == f])) for f in range(5)]
    rowsA.append((nm, np.mean(fs), np.std(fs), f1(y_lab, post(fr_lab, Pm))))
    print(f"   {nm:8s} fold 평균 {np.mean(fs):.4f} ± {np.std(fs):.4f} | 전체 {rowsA[-1][3]:.4f} | fold별 {np.round(fs, 4)}", flush=True)
print("   단계별 F1 (센서 -> 스태커):", np.round(f1_score(y_lab, post(fr_lab, sens_oof), labels=L, average=None), 3), "->",
      np.round(f1_score(y_lab, post(fr_lab, oof2), labels=L, average=None), 3))

# ------------------------------------------------------------ B. 공식 프로토콜
log("B. 공식 프로토콜: 스태커는 Train 행으로만 학습 -> Validation")
trm = (fr_lab.split.values == "train")
vam = ~trm
stk = mk().fit(X_lab[trm], y_lab[trm])
fv_full = fr_lab[vam].reset_index(drop=True)
yv_full = fv_full.label.astype(int).values
cap_idx = F.cap_rows(fv_full).index.values  # fv_full 위치 기준(cap_rows 는 인덱스를 보존)
sens_sources = {"B2 센서=OOF": sens_oof[vam]}
if Path(a.official_model).exists():
    import joblib
    mj = joblib.load(a.official_model)
    mod = mj["model"]
    mod.set_params(n_jobs=a.n_jobs)
    d = pd.concat([pd.read_csv(DATA / "validation" / "validation.csv", dtype={"sample_id": "string", "experiment_id": "string"})], ignore_index=True)
    d["sample_id"] = d.sample_id.astype(str)
    Xs = F.build_features(d)[mj["features"]]
    Pv = pd.DataFrame(mod.predict_proba(Xs), index=d.sample_id.values).loc[fv_full.sample_id.values].values
    sens_sources = {"B1 센서=Train-only 모델(val0.9195)": Pv, **sens_sources}
for nm, Psv in sens_sources.items():
    Piv = img_oof[vam]
    for en, idx in [("cap500", cap_idx), ("full", np.arange(len(fv_full)))]:
        fr = fv_full.iloc[idx].reset_index(drop=True)
        Xv = sfeat(fr, Psv[idx], Piv[idx])
        yv = fr.label.astype(int).values
        base = f1(yv, post(fr, Psv[idx]))
        st = f1(yv, post(fr, stk.predict_proba(Xv)))
        pc0 = f1_score(yv, post(fr, Psv[idx]), labels=L, average=None, zero_division=0)
        pc1 = f1_score(yv, post(fr, stk.predict_proba(Xv)), labels=L, average=None, zero_division=0)
        if en == "cap500" and "head" not in globals():
            neu = np.tile(img_oof.mean(0, keepdims=True), (len(fr), 1))
            head = (st, int((post(fr, stk.predict_proba(Xv)) != post(fr, stk.predict_proba(sfeat(fr, Psv[idx], neu)))).sum()))
        print(f"   [{nm}] {en:6s}: 센서 {base:.4f} -> 스태커 {st:.4f} ({st - base:+.4f}) | 단계별 {np.round(pc0, 3)} -> {np.round(pc1, 3)}", flush=True)

if not a.final:
    sys.exit(0)

# ------------------------------------------------------------ C. 최종 제출
log("C. 최종: 모든 라벨 행으로 스태커 학습 -> Test")
stk_all = mk().fit(X_lab, y_lab)
fr_te = meta.iloc[te].reset_index(drop=True)
Pte = stk_all.predict_proba(sfeat(fr_te, sens_te, img_te))
pred = post(fr_te, Pte)
order = pd.read_csv(DATA / "sample_submission.csv", dtype={"sample_id": "string"})["sample_id"].astype(str).tolist()
s = pd.Series(pred.astype(int), index=fr_te.sample_id.values)
assert set(order) == set(fr_te.sample_id)
sub = pd.DataFrame({"sample_id": order, "label": s.loc[order].values})
assert len(sub) == 3451 and sub.label.isin(L).all() and sub.sample_id.is_unique
out = ROOT / "outputs" / "submissions" / f"{a.name}.csv"
sub.to_csv(out, index=False)
np.save(out.with_suffix(".proba.npy"), Pte.astype(np.float32))
log("저장:", out)
neu_te = np.tile(img_oof.mean(0, keepdims=True), (len(fr_te), 1))
pred_n = post(fr_te, stk_all.predict_proba(sfeat(fr_te, sens_te, neu_te)))
import fuse_common as fc
fc.record(a.name, head[0], head[1], int((pred != pred_n).sum()), out, "stacker")
print("라벨 분포:", sub.label.value_counts(normalize=True).sort_index().round(3).to_dict())
for ref in sorted(out.parent.glob("*.csv")):
    if ref != out and "smoke" not in ref.name:
        o = pd.read_csv(ref, dtype={"sample_id": "string"}).set_index("sample_id").label
        if set(o.index) >= set(sub.sample_id):
            print(f"  {ref.name} 과 일치율 {(o.loc[sub.sample_id.values].values == sub.label.values).mean():.4f}")
