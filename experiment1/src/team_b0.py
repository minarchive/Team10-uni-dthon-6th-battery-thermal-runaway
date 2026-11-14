"""B0: final_lgb 재현(가스 정리 + X1 + 과충전 규칙) + 시간 측정. 결과 OOF 를 outputs/team/b0_*.npy 로 저장."""
import time

import numpy as np

import team_pipeline as T

log = T.log
df = T.load_table()
Lb = df[df.split != "test"].reset_index(drop=True)
log("행", len(df), "라벨", len(Lb), "실험", Lb.experiment_id.nunique(), "| fold 별 실험 수", Lb.groupby("fold").experiment_id.nunique().to_dict())
X = T.get_features(df, "gas_sum", "sum")
lab_i = np.where(df.split != "test")[0]
Xl = X.iloc[lab_i].reset_index(drop=True); y = Lb.label.to_numpy(); fold = Lb.fold.to_numpy(); oc = (Lb.tr_method.values == 2)
tha = T.tha_cummax(Lb)
O = np.zeros((len(y), 6)); Oh = np.zeros((len(y), 6)); times = []
for k in range(5):
    va = fold == k; t0 = time.time()
    m = T.make_lgb().fit(Xl[~va], y[~va]); times.append(time.time() - t0)
    Z = T.proba6(m, Xl[va]); O[va] = Z
    th = T.pick_theta(tha, y, (~va) & oc & np.isin(y, [2, 3]))
    Zh = Z.copy(); vi = np.where(va)[0]; rr = np.where(oc[vi])[0]
    if len(rr): Zh[rr] = T.override(Z[rr], tha[vi[rr]], th)
    Oh[va] = Zh
    log(f"fold {k} 완료 {times[-1]:.0f}s (theta={th:.2f})")
np.save(T.OUT / "b0_oof_nooverride.npy", O); np.save(T.OUT / "b0_oof.npy", Oh)
p0, f0, _ = T.report("B0a 모델만", Lb, y, fold, O)
p1, f1, _ = T.report("B0 모델 + 과충전 규칙 (final_lgb 재현)", Lb, y, fold, Oh, base=p0)
log(f"참고: 서버 보고값 모델만 0.9147 / 규칙 결합 0.9184 | 한 fold 학습 평균 {np.mean(times):.0f}s")
