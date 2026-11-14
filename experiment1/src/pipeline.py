"""학습 파이프라인 함수 모음 (run_train.py 와 노트북이 같이 사용)."""
from __future__ import annotations

import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common as C

N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 4))


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------------------
def prepare(n_aug: int = 2, limit_exp: int = 0):
    """train/val/test 원본 + 이미지 피처 + 피처표. n_aug: 학습 실험의 5/10/15초 간격 증강 사본 수."""
    raw = C.load_all()
    img = pd.concat([C.load_image_cache(s) for s in ["train", "validation", "test"]])
    if limit_exp:  # 스모크 테스트용
        for k in ["train", "validation"]:
            ids = raw[k]["experiment_id"].drop_duplicates().head(limit_exp)
            raw[k] = raw[k][raw[k]["experiment_id"].isin(ids)].reset_index(drop=True)
    out = {"raw": raw, "img": img}
    for k in raw:
        raw[k] = raw[k].reset_index(drop=True)
        out[f"X_{k}"] = C.build_feature_table(raw[k], img)
    out["aug"] = []
    for i in range(n_aug):
        sub = C.subsample_gaps(raw["train"], seed=100 + i).reset_index(drop=True)
        out["aug"].append((sub, C.build_feature_table(sub, img)))
    # val 도 같은 방식으로 간격 섞기 -> '테스트와 비슷한 간격'에서의 강건성 확인용
    vsub = C.subsample_gaps(raw["validation"], seed=7).reset_index(drop=True)
    out["val_gap"] = (vsub, C.build_feature_table(vsub, img))
    return out


def class_weights(y: np.ndarray, mode: str):
    if mode == "none":
        return np.ones(len(y))
    cnt = pd.Series(y).value_counts()
    w = (len(y) / (len(cnt) * cnt)).to_dict()
    if mode == "sqrt":
        w = {k: v ** 0.5 for k, v in w.items()}
    return np.array([w[v] for v in y])


def lgb_params(seed=C.SEED, lr=0.05, ff=0.4):
    return dict(objective="multiclass", num_class=6, learning_rate=lr, num_leaves=31, min_child_samples=60,
                feature_fraction=ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127,
                n_jobs=N_JOBS, seed=seed, verbose=-1, deterministic=True, force_row_wise=True)


def fit(X, y, cw="none", n_est=300, lr=0.05, seed=C.SEED, valid=None, es=0):
    ds = lgb.Dataset(X, label=np.asarray(y) - 1, weight=class_weights(np.asarray(y), cw), free_raw_data=False)
    cbs, vs = [], []
    if valid is not None:
        vs = [lgb.Dataset(valid[0], label=np.asarray(valid[1]) - 1, reference=ds)]
    if es and valid is not None:
        cbs.append(lgb.early_stopping(es, verbose=False))
    return lgb.train(lgb_params(seed, lr), ds, num_boost_round=n_est, valid_sets=vs, callbacks=cbs)


def predict(model, X):
    return model.predict(X, num_iteration=model.best_iteration or None)


def train_matrix(P, cols, use_aug: bool):
    Xs = [P["X_train"][cols]]
    ys = [P["raw"]["train"]["label"].to_numpy()]
    if use_aug:
        for sub, Xa in P["aug"]:
            Xs.append(Xa[cols])
            ys.append(sub["label"].to_numpy())
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


# ---------------------------------------------------------------------------
def feature_sets(cols):
    g = C.feature_groups(cols)
    S = {
        "B0_size_only": g["size"],
        "B0c_cond_only": g["cond"],
        "B0c_cond+size": g["cond"] + g["size"],
        "S1_sensor_raw": g["cond"] + g["sensor_raw"],
        "S2_sensor_causal": g["cond"] + g["sensor_raw"] + g["sensor_causal"],
        "S3_+img_raw": g["cond"] + g["sensor_raw"] + g["sensor_causal"] + g["img_raw"],
        "S4_+img_causal": g["cond"] + g["sensor_raw"] + g["sensor_causal"] + g["img_raw"] + g["img_causal"],
        "S5_+size(shortcut)": g["cond"] + g["sensor_raw"] + g["sensor_causal"] + g["img_raw"] + g["img_causal"] + g["size"],
        "S6_+time": g["cond"] + g["sensor_raw"] + g["sensor_causal"] + g["img_raw"] + g["img_causal"] + g["time"],
        "S7_img_only": g["cond"] + g["img_raw"] + g["img_causal"],
    }
    # size 피처는 img_raw 에서 제외 (size_ 접두사라 이미 분리됨)
    return {k: list(dict.fromkeys(v)) for k, v in S.items()}


def per_class_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=C.LABELS, average=None, zero_division=0)


def eval_val(P, cols, cw="none", use_aug=False, n_est=300, lr=0.05, seed=C.SEED):
    Xtr, ytr = train_matrix(P, cols, use_aug)
    m = fit(Xtr, ytr, cw, n_est, lr, seed)
    yv = P["raw"]["validation"]["label"].to_numpy()
    pv = predict(m, P["X_validation"][cols])
    vg_df, vg_X = P["val_gap"]
    pg = predict(m, vg_X[cols])
    pred = pv.argmax(1) + 1
    return dict(val_f1=C.macro_f1(yv, pred), valgap_f1=C.macro_f1(vg_df["label"].to_numpy(), pg.argmax(1) + 1)), m, pv, per_class_f1(yv, pred)


def cv_trainval(P, cols, cw="none", use_aug=False, n_est=300, lr=0.05, n_splits=5):
    """train+validation 97개 실험을 실험 단위 GroupKFold. 증강은 학습 fold 실험에만 적용."""
    Xall = pd.concat([P["X_train"][cols], P["X_validation"][cols]], ignore_index=True)
    dfall = pd.concat([P["raw"]["train"], P["raw"]["validation"]], ignore_index=True)
    y = dfall["label"].to_numpy()
    g = dfall["experiment_id"].to_numpy()
    oof = np.zeros((len(y), 6))
    f1s = []
    aug_all = None
    if use_aug:
        aug_all = [(pd.concat([P["aug"][i][0]], ignore_index=True), P["aug"][i][1][cols]) for i in range(len(P["aug"]))]
    for tr, va in GroupKFold(n_splits).split(Xall, y, g):
        Xt, yt = Xall.iloc[tr], y[tr]
        if aug_all:
            tr_exp = set(g[tr])
            ex = [(Xa[sub["experiment_id"].isin(tr_exp).to_numpy()], sub["label"].to_numpy()[sub["experiment_id"].isin(tr_exp).to_numpy()])
                  for sub, Xa in aug_all]
            Xt = pd.concat([Xt] + [e[0] for e in ex], ignore_index=True)
            yt = np.concatenate([yt] + [e[1] for e in ex])
        m = fit(Xt, yt, cw, n_est, lr)
        oof[va] = predict(m, Xall.iloc[va])
        f1s.append(C.macro_f1(y[va], oof[va].argmax(1) + 1))
    return float(np.mean(f1s)), float(np.std(f1s)), C.macro_f1(y, oof.argmax(1) + 1), oof, dfall


# ---------------------------------------------------------------------------
# 게이팅 실험용 보조 함수
# ---------------------------------------------------------------------------
def full_matrix(P, cols, use_aug: bool):
    """Train+Validation(+ 간격 증강 사본) 전체 학습 행렬."""
    yv = P["raw"]["validation"]["label"].to_numpy()
    Xs = [pd.concat([P["X_train"][cols], P["X_validation"][cols]], ignore_index=True)]
    ys = [np.concatenate([P["raw"]["train"]["label"].to_numpy(), yv])]
    if use_aug:
        for sub, Xa in P["aug"]:
            Xs.append(Xa[cols]); ys.append(sub["label"].to_numpy())
        for i in range(len(P["aug"])):
            vs = C.subsample_gaps(P["raw"]["validation"], seed=300 + i).reset_index(drop=True)
            Xs.append(C.build_feature_table(vs, P["img"])[cols]); ys.append(vs["label"].to_numpy())
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def fit_binary(X, yb, n_est=250, lr=0.05, seed=C.SEED):
    p = lgb_params(seed, lr)
    p.update(objective="binary")
    p.pop("num_class")
    return lgb.train(p, lgb.Dataset(X, label=np.asarray(yb).astype(int)), num_boost_round=n_est)


def cv_binary(P, cols, thr, n_est=250, lr=0.05, n_splits=5):
    """train+val 실험 단위 GroupKFold OOF 확률 P(y >= thr). cv_trainval 과 같은 fold 분할."""
    Xall = pd.concat([P["X_train"][cols], P["X_validation"][cols]], ignore_index=True)
    dfall = pd.concat([P["raw"]["train"], P["raw"]["validation"]], ignore_index=True)
    y = dfall["label"].to_numpy(); g = dfall["experiment_id"].to_numpy()
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits).split(Xall, y, g):
        m = fit_binary(Xall.iloc[tr], y[tr] >= thr, n_est, lr)
        oof[va] = m.predict(Xall.iloc[va])
    return oof
