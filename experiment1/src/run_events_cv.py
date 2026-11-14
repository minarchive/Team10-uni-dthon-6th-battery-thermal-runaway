"""전처리/사건 피처 효과 검증: 같은 5-fold(실험 단위)에서 A(기존 S2) vs B(+사건 피처) vs C(인과 클리닝 + 사건 피처) 비교."""
import argparse, json
import numpy as np, pandas as pd
import common as C, pipeline as P_, events as E
log = P_.log
ap = argparse.ArgumentParser(); ap.add_argument("--n_est", type=int, default=200); ap.add_argument("--lr", type=float, default=0.08)
a = ap.parse_args()
P = P_.prepare(0)
g = C.feature_groups(list(P["X_train"].columns))
S2 = list(dict.fromkeys(g["cond"] + g["sensor_raw"] + g["sensor_causal"]))
raw = P["raw"]; img = P["img"]
log("사건 피처 계산 (raw / cleaned) ...")
ev, evc, Xc = {}, {}, {}
for k in raw:
    ev[k] = E.event_features(raw[k])
    cl = raw[k].copy(); cl[["atmp_t", "meas_t1", "meas_t2"]] = E.clean_causal(raw[k])[["atmp_t", "meas_t1", "meas_t2"]]
    evc[k] = E.event_features(cl); Xc[k] = C.build_feature_table(cl, img)
    log(k, "cleaned 값 변경:", int((cl[["atmp_t", "meas_t1", "meas_t2"]] != raw[k][["atmp_t", "meas_t1", "meas_t2"]]).any(axis=1).sum()), "프레임")
EVC = list(ev["train"].columns)
variants = {}
for k in ("train", "validation", "test"):
    P[f"B_{k}"] = pd.concat([P[f"X_{k}"], ev[k]], axis=1)
    P[f"C_{k}"] = pd.concat([Xc[k], evc[k]], axis=1)
def run(name, pref, cols):
    Q = dict(P); Q["X_train"], Q["X_validation"], Q["X_test"] = P[f"{pref}_train"], P[f"{pref}_validation"], P[f"{pref}_test"]
    mu, sd, pooled, oof, dfall = P_.cv_trainval(Q, cols, "balanced", False, a.n_est, a.lr)
    y = dfall.label.to_numpy(); isv = (dfall.split == "validation").to_numpy(); pred = oof.argmax(1) + 1
    pc = C.causal_cummax_pred(dfall, pred)
    r = dict(variant=name, n_feat=len(cols), fold_mean=mu, fold_std=sd, pooled=pooled, pooled_cummax=C.macro_f1(y, pc),
             val_subset=C.macro_f1(y[isv], pred[isv]), val_subset_cummax=C.macro_f1(y[isv], pc[isv]))
    pcf = P_.per_class_f1(y, pc); r.update({f"L{i+1}": float(v) for i, v in enumerate(pcf)})
    log(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}); np.save(f"outputs/oof_{name}.npy", oof); return r
rows = [run("A_S2_baseline", "X", S2), run("B_S2+events", "B", S2 + EVC), run("C_clean+events", "C", S2 + EVC)]
pd.DataFrame(rows).to_csv(C.OUT / "events_cv.csv", index=False)
log("저장: outputs/events_cv.csv")
