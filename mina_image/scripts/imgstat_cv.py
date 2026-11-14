"""이미지 '통계'(확률 아님)를 센서 인과 피처 생성기의 추가 신호로 넣어 OOF 5-fold + 공식 분할(Train->Val)로 센서 단독과 비교.

이미지 통계는 experiment1 의 캐시(cache/img_feats_{split}.parquet: 의사색 -> 상대 온도 인덱스 분위수·고온 픽셀 비율·가장자리 등)를 쓴다.
각 통계가 ts_model/features.py 의 SIGNALS 처럼 누적최대·첫 프레임 대비 변화·30/60/120/300초 창(평균·표준편차·최대·최소·기울기·초당 변화)·임계 계산을 모두 받는다.
모두 같은 실험의 현재·과거 관측만 쓴다. 이미지 크기(size_*)는 bat_class 지름길이라 제외한다.
산출: outputs/logs/imgstat_cv.csv, (개선 시) outputs/submissions/<name>.csv, outputs/preds/imgstat_lgb_{oof,test}.npy
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
ap.add_argument("--cache", default="/home/ubuntu/exp1_wt/experiment1/cache")
ap.add_argument("--stats", default="img_mean,img_std,img_max,img_p50,img_p90,img_p99,img_white,img_hot67,img_hot83,img_edge,img_center_minus_all")
ap.add_argument("--n_jobs", type=int, default=16)
ap.add_argument("--fast", action="store_true", help="250 trees lr 0.06 (스크리닝)")
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--name", default="imgstat_lgb")
ap.add_argument("--final", action="store_true", help="OOF 가 센서 단독보다 좋으면 전체 학습 -> Test CSV")
ap.add_argument("--ts_dir", default=str(ROOT / "ts_model"))
a = ap.parse_args()
sys.path.insert(0, a.ts_dir)
import features as F

T0 = time.time()
log = lambda *m: print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *m, flush=True)
L = list(range(1, 7))
STATS = a.stats.split(",")
BASE0 = list(F.BASE_SIGNALS)

tb = table()
frames = []
for s in ["train", "validation", "test"]:
    d = pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    d["sample_id"], d["experiment_id"], d["split"] = d.sample_id.astype(str), d.experiment_id.astype(str), s
    ic = pd.read_parquet(Path(a.cache) / f"img_feats_{s}.parquet")
    ic.index = ic.index.astype(str)
    d = d.merge(ic[STATS].reset_index().rename(columns={"index": "sample_id"}), on="sample_id", how="left")
    frames.append(d)
raw = pd.concat(frames, ignore_index=True).set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
assert raw[STATS].notna().all().all(), "이미지 통계 캐시에 결측"
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab = (raw.label.values >= 1) & (raw.fold.values >= 0)
te = np.where(raw.label.values == -1)[0]
log(f"행 {len(raw)} (라벨 {lab.sum()}, test {len(te)}), 이미지 통계 {len(STATS)}개: {STATS}")


def model():
    m = F.build_model()
    m.set_params(n_jobs=a.n_jobs)
    if a.fast:
        m.set_params(n_estimators=250, learning_rate=0.06)
    return m


def feats(frame, extra):
    F.BASE_SIGNALS = BASE0 + extra
    F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    return F.build_features(frame.reset_index(drop=True))


def f1(y, p):
    return f1_score(y, p, labels=L, average="macro", zero_division=0)


y = raw.label.values.astype(int)
lab_i = np.where(lab)[0]
fr_lab = raw.iloc[lab_i].reset_index(drop=True)
yl = y[lab_i]
fold = fr_lab.fold.values
trm = (fr_lab.split.values == "train")
res, oofs, rows = {}, {}, []
for vn, extra in [("센서 단독", []), ("센서+이미지 통계", STATS)]:
    log(f"[{vn}] feature 계산")
    X = feats(raw, extra)
    Xl = X.iloc[lab_i].reset_index(drop=True)
    log(f"[{vn}] feature {X.shape[1]}개 -> OOF 5-fold")
    O = np.zeros((len(yl), 6))
    for k in range(5):
        va = fold == k
        O[va] = model().fit(Xl[~va], yl[~va]).predict_proba(Xl[va])
        log(f"   fold {k} 완료 F1(후처리) {f1(yl[va], F.postprocess(fr_lab[va].reset_index(drop=True), O[va], a.tau)):.4f}")
    p = F.postprocess(fr_lab, O, a.tau)
    fs = [f1(yl[fold == k], p[fold == k]) for k in range(5)]
    # 공식 분할: Train 행으로 학습 -> Validation(cap500, full)
    m = model().fit(Xl[trm], yl[trm])
    vfull = fr_lab[~trm].reset_index(drop=True)
    cap_idx = F.cap_rows(vfull).index.values
    vcap = vfull.iloc[cap_idx].reset_index(drop=True)
    Pv_full = m.predict_proba(Xl[~trm])
    Pv_cap = m.predict_proba(feats(vcap, extra)[list(Xl.columns)])
    r = dict(variant=vn, n_feat=X.shape[1], oof_f1=f1(yl, p), fold_mean=float(np.mean(fs)), fold_std=float(np.std(fs)), folds=np.round(fs, 4).tolist(),
             val_cap500=f1(vcap.label.astype(int).values, F.postprocess(vcap, Pv_cap, a.tau)),
             val_full=f1(vfull.label.astype(int).values, F.postprocess(vfull, Pv_full, a.tau)),
             per_class=np.round(f1_score(yl, p, labels=L, average=None), 3).tolist())
    rows.append(r)
    oofs[vn] = O
    log(f"[{vn}] OOF F1 {r['oof_f1']:.4f} (fold {r['fold_mean']:.4f}±{r['fold_std']:.4f}) | 공식 Val cap500 {r['val_cap500']:.4f} full {r['val_full']:.4f} | 단계별 {r['per_class']}")
    if vn != "센서 단독" and a.final:
        res["X"], res["extra"], res["cols"] = X, extra, list(Xl.columns)

R = pd.DataFrame(rows)
(ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
R.to_csv(ROOT / "outputs" / "logs" / "imgstat_cv.csv", index=False)
d = R.iloc[1].oof_f1 - R.iloc[0].oof_f1
dv = R.iloc[1].val_cap500 - R.iloc[0].val_cap500
b, c2 = np.array(R.iloc[0].folds), np.array(R.iloc[1].folds)
print(f"\n===== 결과: 이미지 통계 추가 효과 =====\nOOF F1 {R.iloc[0].oof_f1:.4f} -> {R.iloc[1].oof_f1:.4f} ({d:+.4f}) | fold별 차이 {np.round(c2 - b, 4).tolist()} (이긴 fold {(c2 > b).sum()}/5)\n"
      f"공식 Val cap500 {R.iloc[0].val_cap500:.4f} -> {R.iloc[1].val_cap500:.4f} ({dv:+.4f}) | full {R.iloc[0].val_full:.4f} -> {R.iloc[1].val_full:.4f}")
for v in rows:
    if v["oof_f1"] >= 0.85 or v["val_cap500"] >= 0.85:
        print(f"*** 보고 대상 (>=0.85): {v['variant']} OOF {v['oof_f1']:.4f}, Val cap500 {v['val_cap500']:.4f}")
np.save(ROOT / "outputs" / "preds" / "imgstat_lgb_oof.npy", oofs["센서+이미지 통계"].astype(np.float32))
np.save(ROOT / "outputs" / "preds" / "sensor_fast_oof.npy" if a.fast else ROOT / "outputs" / "preds" / "sensor_same_cfg_oof.npy", oofs["센서 단독"].astype(np.float32))

if a.final:
    log("최종: 전체 라벨 행으로 학습 -> Test CSV")
    X, extra, cols = res["X"], res["extra"], res["cols"]
    mf = model().fit(X.iloc[lab_i].reset_index(drop=True)[cols], yl)
    Pt = mf.predict_proba(X.iloc[te].reset_index(drop=True)[cols])
    ft = raw.iloc[te].reset_index(drop=True)
    lab_te = F.postprocess(ft, Pt, a.tau)
    med = raw.iloc[lab_i][STATS].median()
    ft_n = ft.copy()
    ft_n[STATS] = med.values  # 중립 이미지 통계(학습 중앙값 고정)
    Xn = feats(pd.concat([raw.iloc[te][BASE0 + ["experiment_id", "sample_id", "relative_time_s"] + F.META].assign(**{s: med[s] for s in STATS})], ignore_index=True) if False else ft_n, extra)
    lab_n = F.postprocess(ft, mf.predict_proba(Xn[cols]), a.tau)
    order = pd.read_csv(DATA / "sample_submission.csv", dtype={"sample_id": "string"})["sample_id"].astype(str).tolist()
    s = pd.Series(lab_te.astype(int), index=ft.sample_id.values)
    sub = pd.DataFrame({"sample_id": order, "label": s.loc[order].values})
    assert len(sub) == 3451 and sub.label.isin(L).all() and sub.sample_id.is_unique
    out = ROOT / "outputs" / "submissions" / f"{a.name}.csv"
    sub.to_csv(out, index=False)
    np.save(ROOT / "outputs" / "preds" / "imgstat_lgb_test.npy", Pt.astype(np.float32))
    log("저장:", out, "| 라벨 분포", sub.label.value_counts(normalize=True).sort_index().round(3).to_dict(),
        f"| 이미지 통계를 중립값으로 바꾸면 바뀌는 행 {int((lab_te != lab_n).sum())}/3451")
    pd.DataFrame([dict(time=time.strftime("%m-%d %H:%M"), name=a.name, val_f1_cap500=round(R.iloc[1].val_cap500, 4), val_rows_changed_by_image="", test_rows_changed_by_image=int((lab_te != lab_n).sum()),
                       csv=str(out), note=f"imgstat OOF {R.iloc[1].oof_f1:.4f}")]).to_csv(ROOT / "outputs" / "logs" / "candidates.csv", mode="a", header=False, index=False)
