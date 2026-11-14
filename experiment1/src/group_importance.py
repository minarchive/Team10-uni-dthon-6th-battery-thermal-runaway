"""피처 그룹 순열 중요도: 그룹 전체를 같은 순열로 섞어 점수가 얼마나 떨어지는지 (실험 단위 5-fold, 검증 fold 에서만 섞음).
Δ(기준 - 섞은 뒤) <= 0 인 그룹 = 정보 없음/잡음 후보 (가스 제거와 같은 종류의 후보 탐색). 확정은 그룹 제거 재학습으로 따로 확인.
실행: DATA_DIR=... python -u src/group_importance.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common as C
import pipeline as P_

log = P_.log
lab0 = pd.concat([C.load_split(s) for s in ("train", "validation")]).sort_values(["experiment_id", "relative_time_s"], kind="mergesort").reset_index(drop=True)
img = pd.concat([C.load_image_cache(s) for s in ("train", "validation", "test")])
X = C.build_feature_table(lab0, img)
grp = C.feature_groups(list(X.columns))
S2 = list(dict.fromkeys(grp["cond"] + grp["sensor_raw"] + grp["sensor_causal"]))
X = X[S2]
y = lab0.label.to_numpy(); g = lab0.experiment_id.to_numpy()
log(f"프레임 {len(y)}, 실험 {len(set(g))}, 피처 {len(S2)}")

base = lambda c: c.split("__")[0]
kind = lambda c: c.split("__")[1] if "__" in c else "raw"
G = {}
def add(name, cols):
    cols = [c for c in cols if c in S2]
    if cols: G[name] = cols
add("gas (10종+합)", [c for c in S2 if base(c) in C.GAS or base(c) == "gas_sum_log"])
add("atmp_t (주변온도) 원값·윈도", [c for c in S2 if base(c) == "atmp_t"])
add("차이: T1-amb, thermal-amb", [c for c in S2 if base(c) in ("t1_minus_amb", "thermal_minus_amb")])
add("차이: T1-T2, thermal-T1", [c for c in S2 if base(c) in ("t1_minus_t2", "thermal_minus_t1")])
add("indu_t (가열온도)·결측지시", [c for c in S2 if base(c) in ("indu_t", "indu_missing")])
add("압력 meas_p", [c for c in S2 if base(c) == "meas_p"])
add("전압 meas_v", [c for c in S2 if base(c) == "meas_v"])
add("T1 meas_t1", [c for c in S2 if base(c) == "meas_t1"])
add("T2 meas_t2", [c for c in S2 if base(c) == "meas_t2"])
add("thermal_t", [c for c in S2 if base(c) == "thermal_t"])
add("조건 (bat_class, cc, soc, tr)", [c for c in S2 if c in C.COND])
non_gas = [c for c in S2 if base(c) not in C.GAS and base(c) != "gas_sum_log"]
add("윈도 평균 mean30/60", [c for c in non_gas if kind(c).startswith("mean")])
add("윈도 최댓값 max30/60", [c for c in non_gas if kind(c).startswith("max")])
add("시간차 d15/30/60", [c for c in non_gas if kind(c) in ("d15", "d30", "d60")])
add("누적최댓값 cummax", [c for c in non_gas if kind(c) == "cummax"])
add("첫 관측 대비 from_first", [c for c in non_gas if kind(c) == "from_first"])
add("누적최댓값 대비 하락 below_cummax", [c for c in non_gas if kind(c) == "below_cummax"])
add("현재 원값 (윈도 없음)", [c for c in non_gas if kind(c) == "raw" and c not in C.COND])
for k, v in G.items():
    log(f"  {k:36s} {len(v):3d}개")

folds = list(GroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
rng = np.random.default_rng(0)
N_PERM = 3
base_pred = np.zeros(len(y), int)
perm_pred = {k: [np.zeros(len(y), int) for _ in range(N_PERM)] for k in G}
for fi, (tr, va) in enumerate(folds):
    m = P_.fit(X.iloc[tr], y[tr], "balanced", 150, 0.1)
    Xv = X.iloc[va]
    base_pred[va] = P_.predict(m, Xv).argmax(1) + 1
    for k, cols in G.items():
        for r in range(N_PERM):
            Xp = Xv.copy(); idx = rng.permutation(len(Xp)); Xp[cols] = Xv[cols].to_numpy()[idx]
            perm_pred[k][r][va] = P_.predict(m, Xp).argmax(1) + 1
    log(f"fold {fi} 완료")
f0 = C.macro_f1(y, base_pred)
rows = []
for k, cols in G.items():
    fs = [C.macro_f1(y, p) for p in perm_pred[k]]
    # 실험 단위 변동: 실험별 정확도 하락 평균과 부트스트랩 CI
    acc0 = pd.Series(base_pred == y).groupby(g).mean()
    accp = pd.concat([pd.Series(p == y).groupby(g).mean() for p in perm_pred[k]], axis=1).mean(1)
    d = (acc0 - accp).to_numpy(); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    rows.append(dict(group=k, n_feat=len(cols), f1_perm=np.mean(fs), drop=f0 - np.mean(fs), exp_acc_drop=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
R = pd.DataFrame(rows).sort_values("drop")
C.OUT.mkdir(exist_ok=True, parents=True)
R.to_csv(C.OUT / "group_importance.csv", index=False)
log(f"기준 macro-F1 (raw, 전체 길이 풀링) = {f0:.4f}")
print(R.round(4).to_string(index=False), flush=True)
