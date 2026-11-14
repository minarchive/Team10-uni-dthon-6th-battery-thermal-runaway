"""기초 전처리 변형 효과를 OOF 5-fold(fast 설정, 같은 fold)로 비교. 모든 변환은 같은 실험의 현재·과거 값만 쓴다(첫 관측값은 과거).
A clip   : atmp_t 이상값(최대 777°C)을 [10,60] 로 클립
B vratio : meas_v -> meas_v / 실험 첫 관측 meas_v (팩 전압 스케일 제거; 0 이면 0)
C gascap : 가스 10종 / bat_cc (셀 용량으로 정규화)
D pdelta : meas_p -> meas_p - 실험 첫 관측 meas_p
E all    : A+B+C+D + indu_t 제외
각 변형의 개별 효과를 보기 위해 base 와 A~D 를 따로, 마지막에 조합 E 를 돌린다."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7)); BASE0 = list(F.BASE_SIGNALS)
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
print(f"atmp_t > 60°C 행 {(raw.atmp_t > 60).sum()} ({raw[raw.atmp_t > 60].experiment_id.nunique()}개 실험), 단계분포 {raw[raw.atmp_t > 60].label.value_counts().sort_index().to_dict()}", flush=True)
lab_i = np.where((raw.label.values >= 1) & (raw.fold.values >= 0))[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values


def prep(df, opts):
    d = df.copy()
    first = lambda c: d.sort_values("relative_time_s").groupby("experiment_id")[c].transform("first")
    if "A" in opts: d["atmp_t"] = d.atmp_t.clip(10, 60)
    if "B" in opts:
        v0 = first("meas_v"); d["meas_v"] = np.where(v0 > 0, d.meas_v / v0.where(v0 > 0, 1), 0.0)
    if "C" in opts:
        for g in GAS: d[g] = d[g] / d.bat_cc
    if "D" in opts: d["meas_p"] = d.meas_p - first("meas_p")
    return d


def run(name, opts, drop_indu=False):
    d = prep(raw, opts)
    F.BASE_SIGNALS = [s for s in BASE0 if not (drop_indu and s == "indu_t")]
    F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    X = F.build_features(d).iloc[lab_i].reset_index(drop=True)
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k
        m = F.build_model(); m.set_params(n_jobs=16, n_estimators=250, learning_rate=0.06)
        O[va] = m.fit(X[~va], y[~va]).predict_proba(X[va])
    p = F.postprocess(fr, O, 0.7)
    fs = np.array([f1_score(y[fold == k], p[fold == k], labels=L, average="macro", zero_division=0) for k in range(5)])
    return f1_score(y, p, labels=L, average="macro", zero_division=0), fs, f1_score(y, p, labels=L, average=None)

base = None
for name, opts, di in [("base", "", False), ("A clip atmp_t", "A", False), ("B v/v0", "B", False), ("C gas/cc", "C", False), ("D p-p0", "D", False), ("E A+B+C+D+drop indu_t", "ABCD", True), ("F B+C", "BC", False)]:
    f, fs, pc = run(name, opts, di)
    if base is None: base = (f, fs)
    print(f"{name:24s} OOF {f:.4f} ({f - base[0]:+.4f}) | 이긴 fold {int((fs > base[1]).sum())}/5 | fold차이 {np.round(fs - base[1], 4).tolist()} | 단계별 {np.round(pc, 3).tolist()}", flush=True)
