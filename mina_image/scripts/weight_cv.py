"""실험·조건 균형 가중 학습(센서 LightGBM, OOF 5-fold fast, X1 피처 캐시). 가중 = 클래스 균형(기존) x 실험 가중 x 조건 칸 가중.
exp_a: 실험 w ∝ (실험 행 수)^-a  (a=1 이면 실험당 총 가중 동일, a=0.5 절충)   cell_b: 조건 칸(class,cc,soc,tr)별 총 가중을 (칸 총 가중)^-b 로 균형.
지표: 합산 F1(대회와 같은 방식), 실험별 평균 F1, bat_class별 합산 F1, 실험 단위 짝지은 승패."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
meta = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=["experiment_id", "bat_class", "bat_cc", "soc_status", "tr_method"]).drop_duplicates("experiment_id") for s in ["train", "validation"]])
meta["experiment_id"] = meta.experiment_id.astype(str); meta = meta.set_index("experiment_id")
cls = fr.experiment_id.map(meta.bat_class).values
cell = (meta.bat_class.astype(str) + "|" + meta.bat_cc.astype(str) + "|" + meta.soc_status.astype(str) + "|" + meta.tr_method.astype(str)).loc[fr.experiment_id].values
X = pd.read_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet")
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", n_jobs=16, verbose=-1, random_state=42)
nrow = fr.groupby("experiment_id").experiment_id.transform("size").values.astype(float)
cell_tot = pd.Series(1.0, index=range(len(y))).groupby(cell).transform("sum").values
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
eids = fr.experiment_id.values; ue = np.unique(eids); idx = {e: np.where(eids == e)[0] for e in ue}
base_pe = None; base_f = None
for nm, a, b in [("기준(가중 없음)", 0, 0), ("exp a=0.5", 0.5, 0), ("exp a=1.0", 1.0, 0), ("cell b=0.5", 0, 0.5), ("cell b=1.0", 0, 1.0), ("exp0.5+cell0.5", 0.5, 0.5)]:
    w = nrow ** (-a) * cell_tot ** (-b); w = w / w.mean()
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k
        O[va] = lgb.LGBMClassifier(**P0).fit(X[~va], y[~va], sample_weight=w[~va]).predict_proba(X[va])
    p = F.postprocess(fr, O, 0.7)
    pe = np.array([f1(y[idx[e]], p[idx[e]]) for e in ue])
    if base_pe is None: base_pe, base_f = pe, f1(y, p)
    fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    d = pe - base_pe
    print(f"{nm:18s} 합산 OOF {f1(y, p):.4f} ({f1(y, p) - base_f:+.4f}) | 실험별 평균 {pe.mean():.4f} ({d.mean():+.4f}, 이긴 실험 {int((d > 0.001).sum())}/진 실험 {int((d < -0.001).sum())}) | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]} | fold {np.round(fs, 3).tolist()}", flush=True)
