"""PROTOCOL.md 를 자동 실행: 후보 규칙을 순서대로 (그리디, 누적) 평가하고 채택 기준을 코드로 판정한다.

실행: DATA_DIR=... python -u src/protocol_runner.py        (--smoke 로 빠른 점검)
"""
import argparse
import copy
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common as C
import pipeline as P_
import rules as R

log = P_.log
ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--n_est", type=int, default=150)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--final_seeds", type=int, default=3)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--round", type=int, default=1)
a = ap.parse_args()
if a.smoke:
    a.repeats, a.n_est, a.final_seeds = 1, 20, 1
OUT = C.OUT / ("protocol" if a.round == 1 else f"protocol_r{a.round}")
OUT.mkdir(parents=True, exist_ok=True)

# ---- 고정 기준 (PROTOCOL.md 와 동일) -------------------------------------------------
MIN_DELTA, MIN_WINS, VAL_TOL, BOOT = 0.003, 2, -0.01, 2000
if a.smoke:
    MIN_WINS = 1

# ---- 데이터 ---------------------------------------------------------------------------
P = P_.prepare(0, limit_exp=6 if a.smoke else 0)
raw, img = P["raw"], P["img"]
dfall = pd.concat([raw["train"], raw["validation"]], ignore_index=True)
te = raw["test"]
y = dfall.label.to_numpy()
g_all = dfall.experiment_id.to_numpy()
is_val = (dfall.split == "validation").to_numpy()
Xbase_all = pd.concat([P["X_train"], P["X_validation"]], ignore_index=True)
Xbase_te = P["X_test"]
grp = C.feature_groups(list(Xbase_all.columns))
S2 = list(dict.fromkeys(grp["cond"] + grp["sensor_raw"] + grp["sensor_causal"]))  # size_* / img_* 제외
log(f"실험 {dfall.experiment_id.nunique()}개, 프레임 {len(dfall)}, 기준 피처 {len(S2)}개")

# ---- 인과성 검사 ----------------------------------------------------------------------
log("=== 인과성 검사 (앞 60%로 잘라 계산해도 앞부분 값 동일) ===")
causal_ok = {}
causal_ok["BASE"] = R.check_causal(lambda d: C.build_feature_table(d, img)[S2], dfall, n_exp=3)
for k, fn in R.FEATURE_FNS.items():
    causal_ok[k] = R.check_causal(fn, dfall)
log(causal_ok)
assert all(causal_ok.values()), "인과성 검사 실패 -> 중단"

ex_all = {k: fn(dfall) for k, fn in R.FEATURE_FNS.items()}
ex_te = {k: fn(te) for k, fn in R.FEATURE_FNS.items()}

# ---- Test 와 같은 평가: 실험당 500행 초과 시 무작위로 500행(첫 행 포함)만 남기고 피처를 다시 계산 ----------------
CAP = 500


def cap_positions(seed):
    rng = np.random.default_rng(seed)
    t = dfall.relative_time_s.to_numpy(float)
    keep = []
    for _, ix in dfall.groupby("experiment_id", sort=False).indices.items():
        ix = ix[np.argsort(t[ix], kind="mergesort")]
        if len(ix) > CAP:
            sel = np.sort(rng.choice(len(ix) - 1, CAP - 1, replace=False) + 1)
            ix = np.r_[ix[0], ix[sel]]
        keep.append(ix)
    return np.sort(np.concatenate(keep))


CAPD = []
for r in range(a.repeats):
    pos = cap_positions(r)
    dc = dfall.iloc[pos].reset_index(drop=True)
    CAPD.append(dict(pos=pos, df=dc, base=C.build_feature_table(dc, img)[S2], ex={k: fn(dc) for k, fn in R.FEATURE_FNS.items()},
                     y=dc.label.to_numpy(), g=dc.experiment_id.to_numpy(), isv=(dc.split == "validation").to_numpy()))
log(f"Test-like 평가 행수(반복0): {len(CAPD[0]['df'])} (전체 {len(dfall)}); 500행 초과 실험 {int((dfall.groupby('experiment_id').size() > CAP).sum())}개")

# ---- 규칙 상태 / 경계 가중 ----------------------------------------------------------------
grp_all = R.rule_groups(dfall)
grp_te = R.rule_groups(te)
for cd in CAPD:
    cd["grp"] = R.rule_groups(cd["df"])


def boundary_weights(df, width=24):
    w = np.ones(len(df)); t = df.relative_time_s.to_numpy(float); lab_ = df.label.to_numpy()
    for _, ix in df.groupby("experiment_id", sort=False).indices.items():
        ix = ix[np.argsort(t[ix], kind="mergesort")]
        b = np.where(np.diff(lab_[ix]) != 0)[0] + 1
        if len(b):
            near = np.abs(np.arange(len(ix))[:, None] - b[None, :]).min(1) <= width
            w[ix[near]] = 2.0
    return w


bw_all = boundary_weights(dfall)
POST = lambda df, pred, cfg: C.causal_cummax_pred(df, R.apply_rules(df, pred) if cfg.get("post") == "rules" else pred)

# ---- 구성(config) ---------------------------------------------------------------------
BASE = dict(name="BASE", add=[], drop=[], weight="class", model="multi", params={}, dropgas=False, post="cummax")
CANDS1 = [
    ("E1", dict(add=["E1"])),
    ("E2", dict(add=["E2"])),
    ("E3", dict(add=["E3"])),
    ("E4", dict(weight="class+exp")),
    ("E5", dict(drop=["bat_cc", "soc_status"])),
    ("E6", dict(model="ordinal")),
    ("E7", dict(params=dict(num_leaves=15, min_child_samples=200, lambda_l2=20.0))),
]
CANDS2 = [  # PROTOCOL.md 6절 (이벤트 상태기계 중심, 순서 고정)
    ("F1", dict(dropgas=True)),
    ("F2", dict(post="rules")),
    ("F3", dict(add=["F3"])),
    ("F4", dict(model="specialist")),
    ("F5", dict(weight="class+boundary")),
]
CANDS = CANDS1 if a.round == 1 else CANDS2
MONO_PREFIX = ("meas_t1__cummax", "meas_t2__cummax", "thermal_t__cummax", "t1_minus_amb__cummax", "thermal_minus_amb__cummax", "e3_trig_cummax")


def build_X(cfg, which="all"):
    if isinstance(which, int):  # Test-like(500행 상한) 반복 which 의 피처
        base, ex = CAPD[which]["base"], CAPD[which]["ex"]
    else:
        base = Xbase_all if which == "all" else Xbase_te
        ex = ex_all if which == "all" else ex_te
    cols = [c for c in S2 if c not in cfg["drop"]]
    if cfg.get("dropgas"):
        cols = [c for c in cols if c.split("__")[0] not in C.GAS and c.split("__")[0] != "gas_sum_log"]
    parts = [base[cols]] + [ex[k] for k in cfg["add"]]
    return pd.concat(parts, axis=1)


def train_predict(Xtr, ytr, gtr, Xte, cfg, seed=C.SEED, grp_tr=None, grp_te=None):
    p = P_.lgb_params(seed, a.lr)
    p.update(cfg["params"])
    w = P_.class_weights(ytr, "balanced")
    if "boundary" in cfg["weight"]:
        w = w * bw_all[Xtr.index.to_numpy()]
    if cfg["model"] == "specialist":
        probs = np.zeros((len(Xte), 6))
        for gi in (0, 1, 2):
            mtr_, mte_ = grp_tr == gi, grp_te == gi
            if not mte_.any():
                continue
            yg = ytr[mtr_]; cls = np.unique(yg)
            if len(cls) == 0:
                probs[mte_, 0] = 1; continue
            if len(cls) == 1:
                probs[mte_, cls[0] - 1] = 1; continue
            yy = np.searchsorted(cls, yg); p2 = dict(p); p2["num_class"] = len(cls)
            wg = P_.class_weights(yg, "balanced") * (bw_all[Xtr.index.to_numpy()][mtr_] if "boundary" in cfg["weight"] else 1.0)
            m = lgb.train(p2, lgb.Dataset(Xtr[mtr_], label=yy, weight=wg), num_boost_round=a.n_est)
            pr = m.predict(Xte[mte_]).reshape(mte_.sum(), len(cls))
            for j, c_ in enumerate(cls):
                probs[mte_, c_ - 1] = pr[:, j]
        return probs
    if "exp" in cfg["weight"]:
        cnt = pd.Series(gtr).map(pd.Series(gtr).value_counts()).to_numpy()
        w = w / np.sqrt(cnt)
        w = w * len(w) / w.sum()
    if cfg["model"] == "multi":
        m = lgb.train(p, lgb.Dataset(Xtr, label=ytr - 1, weight=w), num_boost_round=a.n_est)
        return m.predict(Xte)
    # 순서형: P(y>=k), k=2..6, 온도 cummax 계열에 단조 증가 제약
    p.pop("num_class")
    p.update(objective="binary", monotone_constraints=[1 if c.startswith(MONO_PREFIX) else 0 for c in Xtr.columns])
    probs = np.zeros((len(Xte), 5))
    for k in range(2, 7):
        yb = (ytr >= k).astype(int)
        wk = P_.class_weights(yb, "balanced")
        if "exp" in cfg["weight"]:
            cnt = pd.Series(gtr).map(pd.Series(gtr).value_counts()).to_numpy()
            wk = wk / np.sqrt(cnt); wk = wk * len(wk) / wk.sum()
        m = lgb.train(p, lgb.Dataset(Xtr, label=yb, weight=wk), num_boost_round=a.n_est)
        probs[:, k - 2] = m.predict(Xte)
    probs = np.minimum.accumulate(probs, axis=1)
    cum = np.c_[np.ones(len(Xte)), probs, np.zeros(len(Xte))]
    return cum[:, :-1] - cum[:, 1:]


def evaluate(cfg):
    """학습은 전체 길이 행, 채점은 Test 와 같은 500행 상한 행(피처 재계산) 에서. 주 지표 = 그 행들의 풀링 Macro-F1(cummax)."""
    X = build_X(cfg)
    res = dict(pooled_cm=[], pooled_raw=[], val_cm=[], acc=[], per_class=[], full_cm=[])
    for r in range(a.repeats):
        cd = CAPD[r]; Xc = build_X(cfg, r)
        oof = np.zeros((len(y), 6)); oofc = np.zeros((len(cd["y"]), 6))
        for tr, va in GroupKFold(5, shuffle=True, random_state=r).split(X, y, g_all):
            Xva = pd.concat([X.iloc[va], Xc[np.isin(cd["g"], np.unique(g_all[va]))]])
            msk = np.isin(cd["g"], np.unique(g_all[va]))
            p = train_predict(X.iloc[tr], y[tr], g_all[tr], Xva, cfg, grp_tr=grp_all[tr], grp_te=np.r_[grp_all[va], cd["grp"][msk]])
            oof[va] = p[:len(va)]; oofc[msk] = p[len(va):]
        pc = POST(cd["df"], oofc.argmax(1) + 1, cfg)
        res["pooled_raw"].append(C.macro_f1(cd["y"], oofc.argmax(1) + 1)); res["pooled_cm"].append(C.macro_f1(cd["y"], pc))
        res["val_cm"].append(C.macro_f1(cd["y"][cd["isv"]], pc[cd["isv"]]))
        res["acc"].append(pd.Series(pc == cd["y"]).groupby(cd["g"]).mean())
        res["per_class"].append(P_.per_class_f1(cd["y"], pc))
        res["full_cm"].append(C.macro_f1(y, POST(dfall, oof.argmax(1) + 1, cfg)))  # 참고: 전체 길이 풀링
    res["acc"] = pd.concat(res["acc"], axis=1)
    return res


def decide(base, cand, cand_id):
    d_primary = float(np.mean(cand["pooled_cm"]) - np.mean(base["pooled_cm"]))
    delta = (cand["acc"].mean(1) - base["acc"].mean(1)).to_numpy()
    rng = np.random.default_rng(0)
    bs = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    wins = int(sum(c > b for c, b in zip(cand["pooled_cm"], base["pooled_cm"])))
    d_val = float(np.mean(cand["val_cm"]) - np.mean(base["val_cm"]))
    checks = {"1_delta>=+0.003": d_primary >= MIN_DELTA, "2_ci_lo>0": bool(lo > 0), f"3_wins>={MIN_WINS}": wins >= MIN_WINS,
              "4_val>=-0.01": d_val >= VAL_TOL, "6_causal": all(causal_ok[k] for k in cand_id_to_fns(cand_id))}
    return dict(delta_primary=d_primary, delta_exp_acc=float(delta.mean()), ci95_lo=float(lo), ci95_hi=float(hi), wins=wins, delta_val=d_val,
                checks=checks, adopt=bool(all(checks.values())), improved_exps=int((delta > 0).sum()), worsened_exps=int((delta < 0).sum()))


def cand_id_to_fns(cid):
    return {"E1": ["E1"], "E2": ["E2"], "E3": ["E3"], "F3": ["F3"]}.get(cid, [])


def merge(cfg, patch, cid):
    c = copy.deepcopy(cfg)
    c["add"] = c["add"] + patch.get("add", [])
    c["drop"] = c["drop"] + patch.get("drop", [])
    c["weight"] = patch.get("weight", c["weight"])
    c["model"] = patch.get("model", c["model"])
    c["params"] = {**c["params"], **patch.get("params", {})}
    c["dropgas"] = patch.get("dropgas", c.get("dropgas", False))
    c["post"] = patch.get("post", c.get("post", "cummax"))
    c["name"] = c["name"] + "+" + cid if c["name"] != "BASE" else cid
    return c


# ---- 그리디 실행 ---------------------------------------------------------------------
log(f"=== 기준(BASE) 평가: {a.repeats}회 반복 x 5-fold ===")
cur_cfg = BASE
cur = evaluate(cur_cfg)
log(f"BASE  [Test-like 500행] pooled(cummax)={np.mean(cur['pooled_cm']):.4f} raw={np.mean(cur['pooled_raw']):.4f} val_subset={np.mean(cur['val_cm']):.4f} "
    f"| 반복별 {np.round(cur['pooled_cm'], 4).tolist()} | 참고 전체길이 풀링 {np.mean(cur['full_cm']):.4f}")
rows = [dict(step="BASE", adopted=True, pooled_cm=np.mean(cur["pooled_cm"]), pooled_raw=np.mean(cur["pooled_raw"]), val_subset=np.mean(cur["val_cm"]))]
decisions = {}
for cid, patch in CANDS:
    cfg = merge(cur_cfg, patch, cid)
    log(f"=== 후보 {cid} 평가 (기준={cur_cfg['name']}) ===")
    cand = evaluate(cfg)
    d = decide(cur, cand, cid)
    decisions[cid] = d
    log(f"{cid}: pooled {np.mean(cand['pooled_cm']):.4f} (Δ{d['delta_primary']:+.4f}) | 실험별 정확도 Δ{d['delta_exp_acc']:+.4f} CI[{d['ci95_lo']:+.4f},{d['ci95_hi']:+.4f}] "
        f"개선{d['improved_exps']}/악화{d['worsened_exps']} | wins {d['wins']}/{a.repeats} | val Δ{d['delta_val']:+.4f}")
    log(f"{cid} 기준 판정: {d['checks']} => {'채택' if d['adopt'] else '기각'}")
    rows.append(dict(step=cid, adopted=d["adopt"], pooled_cm=np.mean(cand["pooled_cm"]), pooled_raw=np.mean(cand["pooled_raw"]), val_subset=np.mean(cand["val_cm"]),
                     delta_primary=d["delta_primary"], delta_exp_acc=d["delta_exp_acc"], ci95_lo=d["ci95_lo"], ci95_hi=d["ci95_hi"], wins=d["wins"], delta_val=d["delta_val"],
                     **{f"L{i+1}": v for i, v in enumerate(np.mean(cand["per_class"], 0))}))
    if d["adopt"]:
        cur_cfg, cur = cfg, cand
        log(f"-> {cid} 채택. 현재 구성: {cur_cfg['name']}")
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    (OUT / "decisions.json").write_text(json.dumps(decisions, indent=2, default=str))

log(f"=== 그리디 종료. 최종 구성: {cur_cfg['name']} ===")

# ---- 최종 점검: 공식 분할, 과적합 격차, 범위 밖 비율, Test 예측 ------------------------------
log("=== 최종 점검 ===")
Xf = build_X(cur_cfg); Xb = build_X(BASE)
mtr = (dfall.split == "train").to_numpy()
def official(cfg, X):
    """공식 분할: Train 만 학습 -> Validation 을 Test 처럼 500행 상한 행(반복0)으로 채점, Train 내 점수와의 격차."""
    cd = CAPD[0]; Xc = build_X(cfg, 0); vm = cd["isv"]
    p_val = train_predict(X[mtr], y[mtr], g_all[mtr], Xc[vm], cfg, grp_tr=grp_all[mtr], grp_te=cd["grp"][vm])
    pv = POST(cd["df"][vm].reset_index(drop=True), p_val.argmax(1) + 1, cfg)
    p_tr = train_predict(X[mtr], y[mtr], g_all[mtr], X[mtr], cfg, grp_tr=grp_all[mtr], grp_te=grp_all[mtr])
    ptr = POST(dfall[mtr].reset_index(drop=True), p_tr.argmax(1) + 1, cfg)
    return C.macro_f1(cd["y"][vm], pv), C.macro_f1(y[mtr], ptr)
vb, tb = official(BASE, Xb); vf, tf = official(cur_cfg, Xf)
log(f"공식 Validation(Train만 학습, cummax): BASE {vb:.4f} (train 내 {tb:.4f}, 격차 {tb - vb:+.4f}) | 최종 {vf:.4f} (train 내 {tf:.4f}, 격차 {tf - vf:+.4f})")
Xte = build_X(cur_cfg, "test")
lo_, hi_ = Xf.min(), Xf.max()
oor = ((Xte < lo_) | (Xte > hi_)).mean()
log(f"Test 피처 중 Train+Val 범위 밖 비율: 평균 {oor.mean():.4f}, 상위 {oor.sort_values(ascending=False).head(5).round(3).to_dict()}")
proba = np.zeros((len(te), 6))
for s in range(a.final_seeds):
    proba += train_predict(Xf, y, g_all, Xte, cur_cfg, seed=C.SEED + s, grp_tr=grp_all, grp_te=grp_te) / a.final_seeds
pred_te = POST(te, proba.argmax(1) + 1, cur_cfg)
ss = pd.read_csv(C.DATA_DIR / "sample_submission.csv", dtype={"sample_id": "string"})
mp = pd.Series(pred_te.astype(int), index=te.sample_id.values)
assert set(ss.sample_id) == set(mp.index)
sub = pd.DataFrame({"sample_id": ss.sample_id.tolist(), "label": mp.loc[ss.sample_id.tolist()].values})
assert sub.label.between(1, 6).all() and len(sub) == len(ss)
fn = "submission_protocol_smoke.csv" if a.smoke else "submission_protocol.csv"
sub.to_csv(OUT / fn, index=False)
log("저장:", OUT / fn, "| 예측 분포", sub.label.value_counts().sort_index().to_dict())
(OUT / "final.json").write_text(json.dumps(dict(final_config=cur_cfg, official_val_base=vb, official_val_final=vf, gap_base=tb - vb, gap_final=tf - vf,
                                                 test_out_of_range_mean=float(oor.mean()), causal=causal_ok), indent=2, default=str))
log("PROTOCOL 완료")
