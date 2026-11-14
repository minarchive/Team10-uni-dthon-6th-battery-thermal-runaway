"""과충전 L3 규칙 비교를 공정하게: (1) 센서 LightGBM P(>=L3)도 같은 LOEO 임계값 조정 (2) 신호·규칙·임계값 선택 전체를 LOEO 안에서(학습 실험으로만) 수행.
oc_l3.py 의 신호 집합(S)을 그대로 쓰고 LightGBM 확률(raw/cummax)을 후보에 추가."""
import runpy, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.argv = ["oc_l3.py", "2"]
ns = runpy.run_path("scripts/oc_l3.py")
S, raw, m23, y, e, ue = ns["S"], ns["raw"], ns["m23"], ns["y"], ns["e"], ns["ue"]
print("\n=== 공정 비교 (과충전) ===")
sp = raw.sens_p3.values.astype(float); cands = {}
eidv = raw.experiment_id.values
for name, s in list(S.items()) + [("lgbm_p3", sp)]:
    s = pd.Series(s); cands[(name, "cummax>=")] = (s.groupby(eidv).cummax().values, 1); cands[(name, "cummin<=")] = (s.groupby(eidv).cummin().values, -1)
    if name == "lgbm_p3": cands[(name, "raw>=")] = (s.values, 1)
def best_th(cm, sign, trm):
    v_ = cm[trm]; v_ = v_[~np.isnan(v_)]
    if len(v_) < 50: return None, -1
    grid = np.unique(np.nanpercentile(v_, np.linspace(2, 98, 25))); b, bt = -1, None
    for th in grid:
        pr = (((cm[trm] >= th) if sign > 0 else (cm[trm] <= th)) & ~np.isnan(cm[trm])).astype(int); f = f1_score(y[trm], pr, zero_division=0)
        if f > b: b, bt = f, th
    return bt, b
# (1) 단일 후보별 LOEO (선택 없음)
for key in [("lgbm_p3", "raw>="), ("lgbm_p3", "cummax>="), ("thermal-amb", "cummax>="), ("thermal_t", "cummax>=")]:
    cm, sign = cands[key]; pred = np.zeros(len(raw), int)
    for ho in ue:
        trm = m23 & (e != ho); tem = m23 & (e == ho); th, _ = best_th(cm, sign, trm)
        pred[tem] = (((cm[tem] >= th) if sign > 0 else (cm[tem] <= th)) & ~np.isnan(cm[tem])).astype(int)
    print(f"  {key[0]:14s} {key[1]:9s} LOEO L3 F1 {f1_score(y[m23], pred[m23], zero_division=0):.4f}")
# (2) 선택까지 LOEO 안에서: 학습 실험에서 (신호,규칙,임계)를 고르고 held-out 에 적용 (센서 LightGBM 후보 제외/포함 둘 다)
for incl in (False, True):
    pred = np.zeros(len(raw), int); picks = []
    for ho in ue:
        trm = m23 & (e != ho); tem = m23 & (e == ho); best = (-1, None, None, None)
        for key, (cm, sign) in cands.items():
            if (not incl) and key[0] == "lgbm_p3": continue
            th, f = best_th(cm, sign, trm)
            if th is not None and f > best[0]: best = (f, key, th, sign)
        f, key, th, sign = best; cm = cands[key][0]; picks.append(key[0])
        pred[tem] = (((cm[tem] >= th) if sign > 0 else (cm[tem] <= th)) & ~np.isnan(cm[tem])).astype(int)
    print(f"  선택까지 LOEO ({'LightGBM 후보 포함' if incl else '센서 신호만'}): L3 F1 {f1_score(y[m23], pred[m23], zero_division=0):.4f} | 고른 신호 빈도 {pd.Series(picks).value_counts().head(4).to_dict()}")
