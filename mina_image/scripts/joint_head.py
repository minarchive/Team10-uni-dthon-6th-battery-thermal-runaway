"""실험 5: 프리즈 임베딩 + 센서 확률 조인트 헤드(MEDIATOR Phase 2 방식, 백본 파인튜닝 없음).

작은 LightGBM 헤드에 입력을 바꿔가며 넣어 무엇이 도움이 되는지 분리한다. 공통 입력 = 센서 확률(log, 확신도, 엔트로피, 기대 단계)과 그 과거 60/180초 평균.
  S      공통만(센서 확률 재보정 기준선)
  S+E    + 프리즈 DINOv2 임베딩 PCA k개 (캐시 features/<enc>/<layer>.npy, 이미지 학습 없음)
  S+I    + 파인튜닝 이미지 확률
  S+E+I  둘 다
PCA 는 라벨 행에서만 적합(비지도). 평가: A) 2단계 5-fold OOF  B) 공식 프로토콜(헤드는 Train 행으로만 학습 -> Validation).
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

import fuse_common as fc
from common import ROOT
from fusion_lib import emb_columns

ap = argparse.ArgumentParser()
ap.add_argument("--img", default="ft_ord_dinov2_b")
ap.add_argument("--emb", default="dinov2_b:b08_mean")
ap.add_argument("--emb_k", type=int, default=16)
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--final", action="store_true")
ap.add_argument("--name", default="joint_head")
ap.add_argument("--n_jobs", type=int, default=6)
ap.add_argument("--ts_dir", default=str(ROOT / "ts_model"))
a = ap.parse_args()
F = fc.load_features_module(a.ts_dir)
c = fc.load(F, a.img, n_jobs=a.n_jobs)
Lr = np.array(fc.L)
lg = lambda p: np.log(np.clip(p, 1e-4, 1))
ent = lambda p: -(p * lg(p)).sum(1)

ids = c.df.sample_id.astype(str).values
fitm = np.zeros(len(ids), bool)
fitm[c.lab_idx] = True
E_all = emb_columns(ids, fitm, a.emb, a.emb_k, 0).values.astype(np.float32)
E_lab = E_all[c.lab_idx]


def roll(fr, vals, name):
    """과거 60/180초 평균(시간 기준, 인과적)."""
    t = fr.relative_time_s.values.astype(float)
    e = fr.experiment_id.values
    o = np.lexsort((t, e))
    tmp = pd.DataFrame({"e": e[o], "ts": pd.to_datetime(t[o], unit="s"), "v": vals[o]})
    out = {}
    for w in (60, 180):
        r = tmp.groupby("e", sort=False).rolling(f"{w}s", on="ts", closed="right")["v"].mean().reset_index(drop=True).values
        col = np.empty(len(fr))
        col[o] = r
        out[f"{name}_m{w}"] = col
    return out


def feats(fr, Ps, Pi, E, use_E, use_I):
    X = {f"ls{k}": lg(Ps)[:, k - 1] for k in fc.L}
    sg3, sg4, ex = Ps[:, 2:].sum(1), Ps[:, 3:].sum(1), Ps @ Lr
    X.update(conf_s=Ps.max(1), ent_s=ent(Ps), exp_s=ex, sg3=sg3, sg4=sg4)
    for nm, v in (("sg3", sg3), ("sg4", sg4), ("exp_s", ex)):
        X.update(roll(fr, v, nm))
    if use_I:
        ig3, ig4 = Pi[:, 2:].sum(1), Pi[:, 3:].sum(1)
        X.update({f"li{k}": lg(Pi)[:, k - 1] for k in fc.L})
        X.update(conf_i=Pi.max(1), ig3=ig3, ig4=ig4, d3=ig3 - sg3, d4=ig4 - sg4)
        X.update(roll(fr, ig3, "ig3"))
    if use_E:
        X.update({f"e{i}": E[:, i] for i in range(E.shape[1])})
    return pd.DataFrame(X)


def mk():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=2.0, class_weight="balanced", random_state=42, n_jobs=a.n_jobs, verbose=-1)


CFG = {"S": (False, False), "S+E": (True, False), "S+I": (False, True), "S+E+I": (True, True)}
y = c.y_lab
fold = c.fr_lab.fold.values
print("\n== A. 2단계 5-fold OOF (센서 단독 argmax 후처리 F1 = %.4f)" % fc.f1(y, fc.post(c, c.fr_lab, c.sens_oof, a.tau)))
for nm, (uE, uI) in CFG.items():
    X = feats(c.fr_lab, c.sens_oof, c.img_oof, E_lab, uE, uI)
    O = np.zeros((len(y), 6))
    for f in range(5):
        va = fold == f
        O[va] = mk().fit(X[~va], y[~va]).predict_proba(X[va])
    fs = [fc.f1(y[fold == f], fc.post(c, c.fr_lab[fold == f], O[fold == f], a.tau)) for f in range(5)]
    print(f"  {nm:6s} fold 평균 {np.mean(fs):.4f} ± {np.std(fs):.4f} | 전체 {fc.f1(y, fc.post(c, c.fr_lab, O, a.tau)):.4f} | 단계별 {fc.pcf1(y, fc.post(c, c.fr_lab, O, a.tau))}", flush=True)

print("\n== B. 공식 프로토콜 (헤드는 Train 행으로만 학습)")
models = {nm: mk().fit(feats(c.fr_lab[c.trm], c.sens_oof[c.trm], c.img_oof[c.trm], E_lab[c.trm], uE, uI), y[c.trm]) for nm, (uE, uI) in CFG.items()}
E_val = E_lab[c.vam]
for vn, fr, Ps, Pi, idx in fc.val_views(c):
    yv = fr.label.astype(int).values
    base = fc.f1(yv, fc.post(c, fr, Ps, a.tau))
    row = [f"센서 {base:.4f}"]
    for nm, (uE, uI) in CFG.items():
        Pb = models[nm].predict_proba(feats(fr, Ps, Pi, E_val[idx], uE, uI))
        row.append(f"{nm} {fc.f1(yv, fc.post(c, fr, Pb, a.tau)):.4f}")
    print(f"  [{vn}] " + " | ".join(row))
    if "cap500" in vn and vn.startswith(next(iter(fc.val_views(c)))[0].split(" | ")[0]):
        VAL = {nm: fc.f1(yv, fc.post(c, fr, models[nm].predict_proba(feats(fr, Ps, Pi, E_val[idx], *CFG[nm])), a.tau)) for nm in CFG}

if a.final:
    cand = {k: v for k, v in VAL.items() if k != "S"}  # 이미지(임베딩/확률)를 쓰는 구성만 제출 후보
    best = max(cand, key=cand.get)
    uE, uI = CFG[best]
    fc.log("최종 구성:", best, "val", round(cand[best], 4))
    Xall = feats(c.fr_lab, c.sens_oof, c.img_oof, E_lab, uE, uI)
    m = mk().fit(Xall, y)
    E_te = E_all[c.te]
    lab_te = fc.post(c, c.fr_te, m.predict_proba(feats(c.fr_te, c.sens_te, c.img_te, E_te, uE, uI)), a.tau)
    neu_E = np.tile(E_lab.mean(0, keepdims=True), (len(c.fr_te), 1))
    lab_n = fc.post(c, c.fr_te, m.predict_proba(feats(c.fr_te, c.sens_te, fc.neutral_img(c, len(c.fr_te)), neu_E, uE, uI)), a.tau)
    # Validation 쪽 이미지 영향(공식 헤드)
    vn0, fr, Ps, Pi, idx = [v for v in fc.val_views(c) if "cap500" in v[0]][0]
    mm = models[best]
    pv = fc.post(c, fr, mm.predict_proba(feats(fr, Ps, Pi, E_val[idx], uE, uI)), a.tau)
    pvn = fc.post(c, fr, mm.predict_proba(feats(fr, Ps, fc.neutral_img(c, len(fr)), np.tile(E_lab.mean(0, keepdims=True), (len(fr), 1)), uE, uI)), a.tau)
    out, sub = fc.write_submission(c, lab_te, a.name)
    print("라벨 분포:", sub.label.value_counts(normalize=True).sort_index().round(3).to_dict())
    fc.record(a.name, cand[best], int((pv != pvn).sum()), int((lab_te != lab_n).sum()), out, f"config={best}")
