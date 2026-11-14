"""학습 분포를 Test 의 그룹(형태x방식) 행 구성에 맞추는 가중치. Test 입력 파일의 조건 열(bat_class, tr_method)과 행 수만 사용(라벨·센서 값 미사용).
w(행) = min(1, 500/실험 행수)  x  mult[그룹], mult[그룹] = (Test 그룹 행 비중) / (학습 그룹 행 비중, 같은 500행 상한 가중), [lo, hi] 로 클립. Test 에 없는 그룹은 lo."""
import numpy as np, pandas as pd
from common import DATA, table
def group_mult(lo=0.2, hi=5.0, power=1.0):
    te = pd.read_csv(DATA / "test" / "test.csv", usecols=["experiment_id", "bat_class", "tr_method"]); tg = te.groupby(["bat_class", "tr_method"]).size(); tg = tg / tg.sum()
    tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0); ids = tb.sample_id.astype(str).values[lab]
    raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=["sample_id", "experiment_id", "bat_class", "tr_method"], dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation"]])
    raw["sample_id"] = raw.sample_id.astype(str); raw = raw.set_index("sample_id").loc[ids].reset_index(); n = raw.groupby("experiment_id").experiment_id.transform("size").values
    w = np.minimum(1.0, 500.0 / n); trg = pd.Series(w).groupby([raw.bat_class.values, raw.tr_method.values]).sum(); trg = trg / trg.sum()
    return {g: float(np.clip((tg.get(g, 0.0) / trg[g]) ** power if tg.get(g, 0.0) > 0 else 0.0, lo, hi)) for g in trg.index}
def row_weights(bat_class, tr_method, experiment_id, mult):
    n = pd.Series(experiment_id).map(pd.Series(experiment_id).value_counts()).values.astype(float)
    return np.minimum(1.0, 500.0 / n) * np.array([mult.get((int(b), int(t)), 1.0) for b, t in zip(bat_class, tr_method)])
if __name__ == "__main__":
    for g, m in sorted(group_mult().items()): print(g, round(m, 2))
