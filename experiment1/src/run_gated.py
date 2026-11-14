"""센서(tabular) 주도 + 이미지 게이팅 보정 실험.

배경: 이미지를 모든 프레임에 피처로 섞으면 성능이 떨어지고(센서 상한 ~0.92), 이미지는 L2/L3/L4 전환 구간에서만 센서를 보완한다.
 -> 센서 확률이 애매(확신도 < tau)하고 센서 예측이 {2,3,4} 인 프레임에서만, 이미지 순서형 확률 P(y>=3), P(y>=4) 를 alpha 비율로 섞는다.
규칙(tau, alpha)은 Validation 이 아니라 train+val 97개 실험 5-fold OOF 로 선택한다.
인과성: 프레임별 확률만 사용(+ 선택적으로 과거 방향 cummax 후처리) -> 미래 정보 없음.
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C
import pipeline as P_

log = P_.log
ap = argparse.ArgumentParser()
ap.add_argument("--n_est", type=int, default=250)
ap.add_argument("--lr", type=float, default=0.06)
ap.add_argument("--final_seeds", type=int, default=3)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.n_est, a.final_seeds = 30, 1
OUT = C.OUT
OUT.mkdir(exist_ok=True, parents=True)

cfg = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
cw, aug = cfg.get("class_weight", "none"), bool(cfg.get("aug", False))
log(f"센서 모델 설정: class_weight={cw}, aug={aug} (run_train 선택값)")
P = P_.prepare(2 if aug else 0, limit_exp=6 if a.smoke else 0)
g = C.feature_groups(list(P["X_train"].columns))
SENS = list(dict.fromkeys(g["cond"] + g["sensor_raw"] + g["sensor_causal"]))
IMG = list(dict.fromkeys(g["cond"] + g["img_raw"] + g["img_causal"]))  # 크기(size_*) 제외: 지름길 차단
dfall = pd.concat([P["raw"]["train"], P["raw"]["validation"]], ignore_index=True)
y = dfall.label.to_numpy()
is_val = (dfall.split == "validation").to_numpy()

# ---- 1) OOF -------------------------------------------------------------------
log("=== 1. 센서 OOF (실험 단위 5-fold) ===")
mu, sd, pooled, Ps, _ = P_.cv_trainval(P, SENS, cw, aug, a.n_est, a.lr)
log(f"센서 OOF: fold mean {mu:.4f}±{sd:.4f} | pooled macro F1 {pooled:.4f} | validation 실험만 {C.macro_f1(y[is_val], Ps[is_val].argmax(1) + 1):.4f}")
log("=== 2. 이미지 순서형 OOF: P(y>=3), P(y>=4) ===")
q3 = P_.cv_binary(P, IMG, 3, a.n_est, a.lr)
q4 = P_.cv_binary(P, IMG, 4, a.n_est, a.lr)
log(f"이미지 AUC y>=3: {roc_auc_score(y >= 3, q3):.3f} | y>=4: {roc_auc_score(y >= 4, q4):.3f}  "
    f"(센서 AUC y>=3: {roc_auc_score(y >= 3, Ps[:, 2:].sum(1)):.3f}, y>=4: {roc_auc_score(y >= 4, Ps[:, 3:].sum(1)):.3f})")
np.save(OUT / "oof_sensor_proba.npy", Ps)
np.save(OUT / "oof_img_q.npy", np.c_[q3, q4])


def gate(Ps, q3, q4, tau, alpha, gset):
    cum = np.cumsum(Ps[:, ::-1], 1)[:, ::-1]  # cum[:, k-1] = P(y>=k)
    ps = Ps.argmax(1) + 1
    conf = Ps.max(1)
    gm = (conf < tau) & np.isin(ps, gset)
    c = cum.copy()
    c[gm, 2] = (1 - alpha) * cum[gm, 2] + alpha * q3[gm]
    c[gm, 3] = (1 - alpha) * cum[gm, 3] + alpha * q4[gm]
    pred = ps.copy()
    pred[gm] = 1 + (c[gm, 1:] > 0.5).sum(1)
    return pred, gm


# ---- 3) 게이팅 규칙 탐색 (OOF 기준) -----------------------------------------------
log("=== 3. 게이팅 규칙 탐색 (선택 기준: 97개 실험 OOF pooled macro F1) ===")
rows = []
for gset, tau, alpha in itertools.product([(2, 3), (2, 3, 4)], [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01], [0, 0.25, 0.5, 0.75, 1.0]):
    pred, gm = gate(Ps, q3, q4, tau, alpha, list(gset))
    pc = C.causal_cummax_pred(dfall, pred)
    rows.append(dict(gset="".join(map(str, gset)), tau=tau, alpha=alpha, gated_frac=gm.mean(), oof_f1=C.macro_f1(y, pred),
                     oof_f1_cummax=C.macro_f1(y, pc), val_subset_f1=C.macro_f1(y[is_val], pred[is_val]),
                     train_subset_f1=C.macro_f1(y[~is_val], pred[~is_val])))
gr = pd.DataFrame(rows)
gr.to_csv(OUT / "gate_grid.csv", index=False)
base = gr[gr.alpha == 0].iloc[0]
best = gr.sort_values("oof_f1", ascending=False).iloc[0]
log(f"센서만(alpha=0): OOF F1 {base.oof_f1:.4f} (val subset {base.val_subset_f1:.4f})")
log(f"최적 게이트: gset={best.gset} tau={best.tau} alpha={best.alpha} gated={best.gated_frac:.3f} -> OOF F1 {best.oof_f1:.4f} "
    f"(val subset {best.val_subset_f1:.4f}, train subset {best.train_subset_f1:.4f})")
print(gr.sort_values("oof_f1", ascending=False).head(10).round(4).to_string(index=False), flush=True)
use_gate = bool(best.oof_f1 > base.oof_f1 + 0.001)
if not use_gate:
    log("게이팅 이득이 0.001 이하 -> 센서 단독을 최종으로 사용")
tau, alpha, gset = (best.tau, best.alpha, [int(c) for c in best.gset]) if use_gate else (1.01, 0.0, [2, 3])

# ---- 4) 오류 분석 --------------------------------------------------------------------
ps = Ps.argmax(1) + 1
pg, gm = gate(Ps, q3, q4, tau, alpha, gset)
ana = []
for L in C.LABELS:
    wrong = (y == L) & (ps != L)
    ana.append(dict(true_label=L, n=int((y == L).sum()), sensor_recall=float((ps[y == L] == L).mean()), sensor_wrong=int(wrong.sum()),
                    wrong_and_gated=int((wrong & gm).sum()), rescued_by_gate=int((wrong & (pg == L)).sum()),
                    broken_by_gate=int(((y == L) & (ps == L) & (pg != L)).sum())))
ana = pd.DataFrame(ana)
ana.to_csv(OUT / "gate_analysis.csv", index=False)
log("라벨별 센서 recall / 게이트로 구제·악화된 프레임:\n" + ana.round(3).to_string(index=False))
pd.DataFrame({"sample_id": dfall.sample_id, "experiment_id": dfall.experiment_id, "relative_time_s": dfall.relative_time_s, "label": y,
              "is_val": is_val, "pred_sensor": ps, "pred_gated": pg, "gated": gm, "q3": q3, "q4": q4}).to_parquet(OUT / "oof_gate.parquet")

# ---- 5) 최종: 전체 재학습 -> Test ------------------------------------------------------
log("=== 5. Train+Val 전체 재학습 -> Test 예측 ===")
te = P["raw"]["test"]
Xs, ys = P_.full_matrix(P, SENS, aug)
n_final = int(a.n_est * 1.15)
Ps_te = np.zeros((len(te), 6))
for s in range(a.final_seeds):
    m = P_.fit(Xs, ys, cw, n_final, a.lr, seed=C.SEED + s)
    Ps_te += P_.predict(m, P["X_test"][SENS]) / a.final_seeds
    log(f"센서 seed {s} 완료")
Xi = pd.concat([P["X_train"][IMG], P["X_validation"][IMG]], ignore_index=True)
q3_te = np.mean([P_.fit_binary(Xi, y >= 3, a.n_est, a.lr, C.SEED + s).predict(P["X_test"][IMG]) for s in range(a.final_seeds)], 0)
q4_te = np.mean([P_.fit_binary(Xi, y >= 4, a.n_est, a.lr, C.SEED + s).predict(P["X_test"][IMG]) for s in range(a.final_seeds)], 0)
pred_te, gm_te = gate(Ps_te, q3_te, q4_te, tau, alpha, gset)
log(f"test 게이트 적용 프레임 {gm_te.mean():.3f}, 센서 대비 변경 {(pred_te != Ps_te.argmax(1) + 1).mean():.3f}")
sel = gr[(gr.gset == "".join(map(str, gset))) & (gr.tau == tau) & (gr.alpha == alpha)].iloc[0]
use_cm = bool(sel.oof_f1_cummax > sel.oof_f1)
log(f"인과 cummax 후처리: OOF {sel.oof_f1:.4f} -> {sel.oof_f1_cummax:.4f} => {'사용' if use_cm else '미사용'}")
if use_cm:
    pred_te = C.causal_cummax_pred(te, pred_te)
np.save(OUT / "test_proba_sensor_gated.npy", Ps_te)
ss = pd.read_csv(C.DATA_DIR / "sample_submission.csv", dtype={"sample_id": "string"})
mp = pd.Series(pred_te.astype(int), index=te.sample_id.values)
assert set(ss.sample_id) == set(mp.index)
sub = pd.DataFrame({"sample_id": ss.sample_id.tolist(), "label": mp.loc[ss.sample_id.tolist()].values})
assert sub.label.between(1, 6).all() and len(sub) == len(ss)
fn = "submission_gated_smoke.csv" if a.smoke else "submission_gated.csv"
sub.to_csv(OUT / fn, index=False)
log("저장:", OUT / fn, "| 예측 분포", sub.label.value_counts().sort_index().to_dict())
(OUT / "gated_summary.json").write_text(json.dumps(dict(
    use_gate=use_gate, tau=float(tau), alpha=float(alpha), gset=gset, cummax=use_cm,
    sensor_only_oof_f1=float(base.oof_f1), gated_oof_f1=float(sel.oof_f1_cummax if use_cm else sel.oof_f1),
    sensor_only_val_subset=float(base.val_subset_f1), gated_val_subset=float(sel.val_subset_f1),
    test_gated_frac=float(gm_te.mean()), test_dist={int(k): int(v) for k, v in sub.label.value_counts().sort_index().items()}), indent=2))
log("GATED SUMMARY 저장 완료")
