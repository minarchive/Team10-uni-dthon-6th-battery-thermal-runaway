"""cum_route.py / joint_head.py 공용: 센서·이미지 OOF 확률과 공식 프로토콜(Train->Validation) 평가 틀.

ctx 구성(모두 folds.parquet 의 라벨 행 순서):
  fr_lab, y_lab            라벨 행의 메타(experiment_id, relative_time_s, split, fold)와 라벨
  sens_oof, img_oof        센서(sensor_oof.py)·이미지(finetune.py) 5-fold OOF 확률 (n,6)
  sens_te, img_te          Test 확률(fold 평균)
  trm, vam                 Train / Validation 행 마스크(라벨 행 기준)
  fv_full, cap_idx         Validation 프레임(full)과 cap500 에 해당하는 위치
  Psv_official             val0.9195(Train-only) 센서 모델의 Validation 확률(fv_full 순서). 모델 파일이 없으면 None
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import DATA, ROOT, table

L = list(range(1, 7))
P = ROOT / "outputs" / "preds"
T0 = time.time()


def log(*m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *m, flush=True)


def load_features_module(ts_dir):
    sys.path.insert(0, str(ts_dir))
    import features as F
    return F


def load(F, img="ft_ord_dinov2_b", official_model="/home/ubuntu/nicole/val0.9195/model.joblib", n_jobs=6):
    df = table()
    y_all, fold_all = df.label.values, df.fold.values
    lab = (y_all >= 1) & (fold_all >= 0)
    lab_idx, te = np.where(lab)[0], np.where(y_all == -1)[0]
    tags = img.split(",")
    c = SimpleNamespace(df=df, lab_idx=lab_idx, te=te, F=F, tags=tags)
    c.sens_oof, c.sens_te = np.load(P / "sensor_lgb_oof.npy"), np.load(P / "sensor_lgb_test.npy")
    c.img_oof = np.mean([np.load(P / f"{t}_oof.npy") for t in tags], 0)
    c.img_te = np.mean([np.load(P / f"{t}_test.npy") for t in tags], 0)
    meta = df[["sample_id", "experiment_id", "relative_time_s", "split", "label", "fold"]].copy()
    meta["experiment_id"], meta["sample_id"] = meta.experiment_id.astype(str), meta.sample_id.astype(str)
    c.meta = meta
    c.fr_lab = meta.iloc[lab_idx].reset_index(drop=True)
    c.fr_te = meta.iloc[te].reset_index(drop=True)
    c.y_lab = c.fr_lab.label.astype(int).values
    c.trm = (c.fr_lab.split.values == "train")
    c.vam = ~c.trm
    c.fv_full = c.fr_lab[c.vam].reset_index(drop=True)
    c.cap_idx = F.cap_rows(c.fv_full).index.values
    c.Psv_official = None
    if Path(official_model).exists():
        import joblib
        mj = joblib.load(official_model)
        mod = mj["model"]
        mod.set_params(n_jobs=n_jobs)
        d = pd.read_csv(DATA / "validation" / "validation.csv", dtype={"sample_id": "string", "experiment_id": "string"})
        d["sample_id"] = d.sample_id.astype(str)
        Pv = pd.DataFrame(mod.predict_proba(F.build_features(d)[mj["features"]]), index=d.sample_id.values)
        c.Psv_official = Pv.loc[c.fv_full.sample_id.values].values
    log(f"라벨 행 {len(lab_idx)} (Train {c.trm.sum()} / Val {c.vam.sum()}), test {len(te)}, 이미지={tags}, 공식 센서 모델={'있음' if c.Psv_official is not None else '없음'}")
    return c


def f1(y, p):
    return f1_score(y, p, labels=L, average="macro", zero_division=0)


def pcf1(y, p):
    return np.round(f1_score(y, p, labels=L, average=None, zero_division=0), 3).tolist()


def post(c, fr, Pm, tau=0.7):
    return c.F.postprocess(fr.reset_index(drop=True), Pm, tau)


def val_views(c):
    """공식 프로토콜 평가 단위: (이름, Validation 프레임, 센서확률, 이미지확률, 위치)."""
    sens = {"센서=OOF": c.sens_oof[c.vam]}
    if c.Psv_official is not None:
        sens = {"센서=Train-only(val0.9195)": c.Psv_official, **sens}
    out = []
    for nm, Ps in sens.items():
        for en, idx in [("cap500", c.cap_idx), ("full", np.arange(len(c.fv_full)))]:
            out.append((f"{nm} | {en}", c.fv_full.iloc[idx].reset_index(drop=True), Ps[idx], c.img_oof[c.vam][idx], idx))
    return out


def neutral_img(c, n):
    """'빈 이미지' 대용: 학습 OOF 이미지 확률의 평균 행을 n 행 복제. 실제 빈 이미지 추론은 체크포인트가 없어 불가(근사)."""
    return np.tile(c.img_oof.mean(0, keepdims=True), (n, 1))


def write_submission(c, labels, name):
    order = pd.read_csv(DATA / "sample_submission.csv", dtype={"sample_id": "string"})["sample_id"].astype(str).tolist()
    s = pd.Series(np.asarray(labels).astype(int), index=c.fr_te.sample_id.values)
    assert set(order) == set(s.index)
    sub = pd.DataFrame({"sample_id": order, "label": s.loc[order].values})
    assert len(sub) == 3451 and sub.label.isin(L).all() and sub.sample_id.is_unique
    out = ROOT / "outputs" / "submissions" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    return out, sub


def record(name, val_f1, val_changed, test_changed, csv, note=""):
    """후보 한 줄 기록 + 검증 F1 >= 0.85 이면 눈에 띄게 보고."""
    row = pd.DataFrame([dict(time=time.strftime("%m-%d %H:%M"), name=name, val_f1_cap500=round(val_f1, 4), val_rows_changed_by_image=val_changed,
                             test_rows_changed_by_image=test_changed, csv=str(csv), note=note)])
    f = ROOT / "outputs" / "logs" / "candidates.csv"
    row.to_csv(f, mode="a", header=not f.exists(), index=False)
    print(f"\nCANDIDATE {name}: Validation Macro F1(cap500, 공식) {val_f1:.4f} | 이미지를 중립값으로 바꾸면 바뀌는 행: Val {val_changed}, Test {test_changed}/3451 | {csv}", flush=True)
    if val_f1 >= 0.85:
        print("*** 보고 대상 (Validation >= 0.85) ***", name, round(val_f1, 4), flush=True)
