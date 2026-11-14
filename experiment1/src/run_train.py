"""전체 학습 파이프라인 (ablation -> 설정 선택 -> 후처리 선택 -> train+val 재학습 -> submission.csv).

선택 기준은 모두 'Validation Macro F1'. 실행 로그는 stdout 으로 바로바로 출력(-u).
예) DATA_DIR=/home/ubuntu/data/battery_data_light python -u src/run_train.py
"""
import argparse
import json

import numpy as np
import pandas as pd

import common as C
import pipeline as P_

log = P_.log
ap = argparse.ArgumentParser()
ap.add_argument("--n_aug", type=int, default=2)
ap.add_argument("--n_est", type=int, default=250, help="ablation/그리드용 트리 수")
ap.add_argument("--lr", type=float, default=0.08)
ap.add_argument("--final_seeds", type=int, default=3)
ap.add_argument("--no_cv", action="store_true", help="GroupKFold CV 생략(빠른 실행)")
ap.add_argument("--smoke", action="store_true", help="스모크 테스트: 실험 6개만, 트리 30개")
a = ap.parse_args()
if a.smoke:
    a.n_est, a.n_aug, a.final_seeds = 30, 1, 1
C.OUT.mkdir(exist_ok=True, parents=True)

log("DATA_DIR =", C.DATA_DIR, "| n_jobs =", P_.N_JOBS)
log("피처 생성 (이미지 캐시 + 인과 시계열) ...")
P = P_.prepare(a.n_aug, limit_exp=6 if a.smoke else 0)
yv = P["raw"]["validation"]["label"].to_numpy()
log("train", P["X_train"].shape, "val", P["X_validation"].shape, "test", P["X_test"].shape, "| aug 사본", len(P["aug"]))
cols_all = list(P["X_train"].columns)
FS = P_.feature_sets(cols_all)

# --- 1. feature-set ablation -------------------------------------------------
log("=== 1. Feature-set ablation (class_weight=none, aug 없음) ===")
rows, val_probs = [], {}
for name, cols in FS.items():
    res, m, pv, pcf = P_.eval_val(P, cols, "none", False, a.n_est, a.lr)
    row = dict(feature_set=name, n_feat=len(cols), **res)
    for i, v in enumerate(pcf, 1):
        row[f"f1_L{i}"] = v
    if not a.no_cv and not a.smoke:
        mu, sd, pooled, _, _ = P_.cv_trainval(P, cols, "none", False, a.n_est, a.lr)
        row.update(cv_f1_mean=mu, cv_f1_std=sd, cv_f1_pooled=pooled)
    rows.append(row)
    val_probs[name] = pv
    log(f"{name:22s} n={len(cols):4d} val_F1={res['val_f1']:.4f} valgap_F1={res['valgap_f1']:.4f}" +
        (f" | cv={row.get('cv_f1_mean', float('nan')):.4f}±{row.get('cv_f1_std', float('nan')):.4f}" if 'cv_f1_mean' in row else ""))
abl = pd.DataFrame(rows)
abl.to_csv(C.OUT / "ablation.csv", index=False)
# B0/B0c 는 기준선이므로 최종 후보에서 제외
cand = abl[~abl.feature_set.str.startswith("B0")]
best_fs = cand.sort_values("val_f1", ascending=False).iloc[0].feature_set
log("선택된 feature set:", best_fs, "| B0 대비 이득 확인: B0_size_only val_F1 =", round(float(abl.set_index('feature_set').loc['B0_size_only', 'val_f1']), 4))
cols = FS[best_fs]

# --- 2. class_weight x 간격 증강 그리드 -----------------------------------------
log("=== 2. class_weight x gap-augmentation 그리드 ===")
grid = []
for cw in ["none", "sqrt", "balanced"]:
    for aug in ([False, True] if P["aug"] else [False]):
        res, m, pv, pcf = P_.eval_val(P, cols, cw, aug, a.n_est, a.lr)
        grid.append(dict(class_weight=cw, aug=aug, **res))
        val_probs[f"{cw}|aug={aug}"] = pv
        log(f"cw={cw:8s} aug={aug!s:5s} val_F1={res['val_f1']:.4f} valgap_F1={res['valgap_f1']:.4f}")
gdf = pd.DataFrame(grid)
gdf.to_csv(C.OUT / "grid.csv", index=False)
bg = gdf.sort_values("val_f1", ascending=False).iloc[0]
cw, aug = bg.class_weight, bool(bg.aug)
log(f"선택: class_weight={cw}, aug={aug}, val_F1={bg.val_f1:.4f}")

# --- 3. 후처리(인과) 선택 ------------------------------------------------------
log("=== 3. 인과 후처리 선택 ===")
pv = val_probs[f"{cw}|aug={aug}"]
pred0 = pv.argmax(1) + 1
vdf = P["raw"]["validation"]
posts = {"raw": pred0, "cummax": C.causal_cummax_pred(vdf, pred0)}
for k in (3, 5, 9, 15):
    posts[f"vote{k}_mono"] = C.causal_vote_pred(vdf, pred0, k, True)
    posts[f"vote{k}"] = C.causal_vote_pred(vdf, pred0, k, False)
prow = [dict(post=k, val_f1=C.macro_f1(yv, v)) for k, v in posts.items()]
pdf = pd.DataFrame(prow)
pdf.to_csv(C.OUT / "postproc.csv", index=False)
for r in prow:
    log(f"post={r['post']:12s} val_F1={r['val_f1']:.4f}")
best_post = pdf.sort_values("val_f1", ascending=False).iloc[0].post
log("선택된 후처리:", best_post)
final_val_pred = posts[best_post]
val_f1_final = C.macro_f1(yv, final_val_pred)
from sklearn.metrics import classification_report, confusion_matrix
log("Validation 최종(train만 학습) Macro F1 =", round(val_f1_final, 4))
print(classification_report(yv, final_val_pred, digits=4, zero_division=0), flush=True)
print("confusion matrix (행=정답, 열=예측):\n", confusion_matrix(yv, final_val_pred, labels=C.LABELS), flush=True)
pd.DataFrame({"sample_id": vdf.sample_id, "experiment_id": vdf.experiment_id, "relative_time_s": vdf.relative_time_s,
              "label": yv, "pred_raw": pred0, "pred_final": final_val_pred}).to_parquet(C.OUT / "val_pred.parquet")
np.save(C.OUT / "val_proba.npy", pv)

# --- 4. best_iter (validation early-stopping) -------------------------------------
log("=== 4. early stopping 으로 트리 수 결정 ===")
Xtr, ytr = P_.train_matrix(P, cols, aug)
m_es = P_.fit(Xtr, ytr, cw, 2000 if not a.smoke else 60, 0.03, valid=(P["X_validation"][cols], yv), es=100)
best_iter = int(m_es.best_iteration or m_es.current_iteration())
log("best_iter =", best_iter)
imp = pd.DataFrame({"feature": cols, "gain": m_es.feature_importance("gain")}).sort_values("gain", ascending=False)
imp.to_csv(C.OUT / "feature_importance.csv", index=False)
log("상위 15 중요도:\n" + imp.head(15).to_string(index=False))

# --- 5. 최종: Train+Validation 재학습 -> Test 예측 --------------------------------
log("=== 5. Train+Validation 재학습 후 Test 예측 ===")
Xs = [pd.concat([P["X_train"][cols], P["X_validation"][cols]], ignore_index=True)]
ys = [np.concatenate([P["raw"]["train"]["label"].to_numpy(), yv])]
if aug:
    for sub, Xa in P["aug"]:
        Xs.append(Xa[cols]); ys.append(sub["label"].to_numpy())
    # 검증 실험도 간격 증강 사본 추가
    for i in range(len(P["aug"])):
        vs = C.subsample_gaps(P["raw"]["validation"], seed=300 + i).reset_index(drop=True)
        Xs.append(C.build_feature_table(vs, P["img"])[cols]); ys.append(vs["label"].to_numpy())
Xfull, yfull = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
n_final = max(30, int(best_iter * 1.15))
te = P["raw"]["test"]
proba = np.zeros((len(te), 6))
for s in range(a.final_seeds):
    m = P_.fit(Xfull, yfull, cw, n_final, 0.03, seed=C.SEED + s)
    proba += P_.predict(m, P["X_test"][cols]) / a.final_seeds
    log(f"seed {s} 완료 (n_estimators={n_final}, 학습행 {len(Xfull)})")
pred_te = proba.argmax(1) + 1
pred_te = {"raw": pred_te, "cummax": C.causal_cummax_pred(te, pred_te)}.get(best_post) if best_post in ("raw", "cummax") \
    else C.causal_vote_pred(te, pred_te, int(best_post.split("vote")[1].split("_")[0]), best_post.endswith("_mono"))
np.save(C.OUT / "test_proba.npy", proba)

sub_t = pd.read_csv(C.DATA_DIR / "sample_submission.csv", dtype={"sample_id": "string"})
mp = pd.Series(pred_te.astype(int), index=te.sample_id.values)
assert set(sub_t.sample_id) == set(mp.index), "sample_submission 과 test 의 ID 불일치"
sub = pd.DataFrame({"sample_id": sub_t.sample_id.tolist(), "label": mp.loc[sub_t.sample_id.tolist()].values})
assert sub.label.isin(C.LABELS).all() and len(sub) == len(sub_t)
if a.smoke:
    sub.to_csv(C.OUT / "submission_smoke.csv", index=False)
else:
    sub.to_csv(C.OUT / "submission.csv", index=False)
log("저장 완료:", C.OUT / ("submission_smoke.csv" if a.smoke else "submission.csv"), "| 예측 분포:", sub.label.value_counts().sort_index().to_dict())

summary = dict(best_feature_set=best_fs, class_weight=cw, aug=aug, post=best_post, best_iter=best_iter, n_final_estimators=n_final,
               val_macro_f1_train_only=val_f1_final, test_pred_dist=sub.label.value_counts().sort_index().to_dict())
(C.OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
log("SUMMARY", json.dumps(summary, default=str))
