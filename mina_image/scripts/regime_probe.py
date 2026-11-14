"""L3 시작 레짐(A: 전환 시 meas_t1<65°C / B: >=65°C)을 L2 시작 직후 정보로 미리 예측할 수 있는가? (실험 단위 leave-one-out)

특징은 '메타'(bat_cc/class/soc/tr_method), '센서'(L2 시작 후 H초까지의 meas_t1 기울기·압력 상승·가스·전압비), '이미지'(같은 구간 프리즈 임베딩 평균의 PCA),
조합으로 AUC/정확도를 낸다. L2 시작은 라벨이 아니라 'meas_t1 누적최대가 처음 45°C 이상' 규칙(features.py 근거: Train/Val 100% 일치)으로 잡는다.
H 이전의 관측만 쓴다(미래 정보 없음). 레짐 정답(L3 시작 시점 meas_t1)은 학습용 라벨로만 쓴다.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, ROOT, table

H_LIST = [30, 45, 60]
t = table()
lab = (t.label.values >= 1) & (t.fold.values >= 0)
E = np.load(ROOT / "features" / "dinov2_b" / "b08_mean.npy").astype(np.float32)
pos = pd.Series(np.arange(len(t)), index=t.sample_id.astype(str).values)
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation"]])
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw["emb_row"] = pos.loc[raw.sample_id].values
Z = PCA(8, random_state=0).fit_transform((E[raw.emb_row.values] - E[raw.emb_row.values].mean(0)) / (E[raw.emb_row.values].std(0) + 1e-6))
raw[[f"z{i}" for i in range(8)]] = Z
gas = [c for c in raw.columns if c.startswith("meas_") and c not in ("meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p", "meas_t1_plus_t2")]
raw["gas"] = raw[gas].sum(axis=1)
rows = []
for e, g in raw.sort_values(["experiment_id", "relative_time_s"]).groupby("experiment_id"):
    g = g.reset_index(drop=True)
    i3 = g.index[g.label >= 3]
    cm = g.meas_t1.cummax()
    i45 = g.index[cm >= 45]
    if len(i3) == 0 or len(i45) == 0:
        continue
    a, j = i45[0], i3[0]
    r = dict(exp=e, regime=int(g.meas_t1.iloc[j] >= 65), cc=g.bat_cc.iloc[0], cls=g.bat_class.iloc[0], soc=g.soc_status.iloc[0], trm=g.tr_method.iloc[0])
    for h in H_LIST:
        w = g[(g.relative_time_s >= g.relative_time_s.iloc[a]) & (g.relative_time_s <= g.relative_time_s.iloc[a] + h)]
        w = w[w.index < j] if len(w[w.index < j]) >= 3 else w.iloc[:3]  # L3 시작 이전 관측만(사후 라벨은 구간 자르는 데만 사용, 구간이 너무 짧으면 처음 3행)
        dt = max(w.relative_time_s.iloc[-1] - w.relative_time_s.iloc[0], 1.0)
        r.update({f"t1_slope_{h}": (w.meas_t1.iloc[-1] - w.meas_t1.iloc[0]) / dt, f"p_rise_{h}": w.meas_p.iloc[-1] - w.meas_p.iloc[0],
                  f"gas_{h}": w.gas.iloc[-1], f"t2_t1_{h}": w.meas_t2.iloc[-1] - w.meas_t1.iloc[-1], f"v_{h}": w.meas_v.iloc[-1] / max(g.meas_v.iloc[0], 1e-6)})
        for i in range(8):
            r[f"z{i}_{h}"] = w[f"z{i}"].mean()
    rows.append(r)
d = pd.DataFrame(rows)
y = d.regime.values
print(f"실험 {len(d)}개, 레짐 B 비율 {y.mean():.2f} (다수결 정확도 {max(y.mean(), 1 - y.mean()):.3f})")
meta = pd.get_dummies(d[["cc", "cls", "soc", "trm"]].astype(str)).astype(float)
for h in H_LIST:
    sens = d[[f"{k}_{h}" for k in ("t1_slope", "p_rise", "gas", "t2_t1", "v")]]
    img = d[[f"z{i}_{h}" for i in range(8)]]
    sets = {"메타": meta, "센서": sens, "이미지": img, "센서+이미지": pd.concat([sens, img], axis=1), "메타+센서+이미지": pd.concat([meta, sens, img], axis=1)}
    print(f"\n[L2 시작 후 {h}초까지 관측]")
    for nm, X in sets.items():
        out = []
        for mn, m in [("로지스틱", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.3))), ("RF", RandomForestClassifier(300, min_samples_leaf=3, random_state=0))]:
            p = cross_val_predict(m, X.fillna(0).values, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
            out.append(f"{mn} AUC {roc_auc_score(y, p):.3f} 정확도 {((p > .5) == y).mean():.3f}")
        print(f"  {nm:12s} " + " | ".join(out))
