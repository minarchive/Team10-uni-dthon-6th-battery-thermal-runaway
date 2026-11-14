"""트리거 후보 신호의 '임계 도달 후 경과 시간' 피처를 추가해 OOF 5-fold(fast)로 효과 검증. 모두 과거 관측만 사용(누적 극값, 첫 관측 대비).
X0 base / X1 +thermal_t(절대·상대) 임계 / X2 +thermal+t1 상대+압력 상대+가스합 임계. 과충전/가열 그룹별 점수도 본다."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab_i = np.where((raw.label.values >= 1) & (raw.fold.values >= 0))[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values
trm = fr.tr_method.values


def cross_feats(df, spec):
    """spec: {이름: (시리즈 정의 함수, [레벨...])}. 반환: 입력 행 순서 DataFrame (경과 시간, 도달 전 NaN)."""
    d = df.sort_values(["experiment_id", "relative_time_s"])
    eid, t = d.experiment_id, d.relative_time_s
    g = d.groupby("experiment_id")
    out = {}
    for name, (fn, levels) in spec.items():
        s = fn(d, g)
        cm = s.groupby(eid).cummax()
        for lv in levels:
            tf = t.where(cm >= lv).groupby(eid).cummin()
            out[f"cx_{name}_{lv}"] = (t - tf)
    return pd.DataFrame(out).loc[df.index]


gsum = lambda d: d[GAS].sum(axis=1)
SPEC1 = {"thermal": (lambda d, g: d.thermal_t, [35, 40, 45, 50, 55, 60, 65, 70, 80, 100, 150, 200]),
         "thermal_rel": (lambda d, g: d.thermal_t - g.thermal_t.transform("first"), [2, 5, 10, 15, 20, 30, 40])}
SPEC2 = {**SPEC1,
         "t1_rel": (lambda d, g: d.meas_t1 - g.meas_t1.transform("first"), [10, 20, 30, 40, 50]),
         "t1_amb": (lambda d, g: d.meas_t1 - d.atmp_t, [20, 30, 40, 50]),
         "p_rel": (lambda d, g: d.meas_p - g.meas_p.transform("first"), [0.5, 1, 2, 5, 10, 15, 20, 30]),
         "gas": (lambda d, g: gsum(d), [1, 2, 5, 10, 20, 40])}
base = F.build_features(raw)
X0 = base
variants = {"X0 base": X0, "X1 +thermal 임계": pd.concat([base, cross_feats(raw, SPEC1).set_index(base.index)], axis=1),
            "X2 +thermal+t1상대+압력+가스 임계": pd.concat([base, cross_feats(raw, SPEC2).set_index(base.index)], axis=1)}
res = {}
for nm, X in variants.items():
    Xl = X.iloc[lab_i].reset_index(drop=True)
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k
        m = F.build_model(); m.set_params(n_jobs=16, n_estimators=250, learning_rate=0.06)
        O[va] = m.fit(Xl[~va], y[~va]).predict_proba(Xl[va])
    p = F.postprocess(fr, O, 0.7)
    fs = np.array([f1_score(y[fold == k], p[fold == k], labels=L, average="macro", zero_division=0) for k in range(5)])
    f = lambda m: f1_score(y[m], p[m], labels=L, average="macro", zero_division=0)
    res[nm] = (f(np.ones(len(y), bool)), fs)
    b = res["X0 base"]
    print(f"{nm:34s} feat {X.shape[1]} | OOF {res[nm][0]:.4f} ({res[nm][0] - b[0]:+.4f}) 이긴 fold {int((fs > b[1]).sum())}/5 | 가열 {f(trm == 1):.4f} 과충전 {f(trm == 2):.4f} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
