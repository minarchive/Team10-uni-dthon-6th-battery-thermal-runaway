"""센서 LightGBM 하이퍼파라미터 축별 탐색 (OOF 5-fold, 같은 fold, X1=base+thermal 임계 피처). 선택은 OOF 로만.
한 번에 한 축만 바꿔 기준(C0)과 fold 별로 비교한다. 결과 OOF 확률은 outputs/preds/tune_<cfg>_oof.npy 에 저장."""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
lab_i = np.where((raw.label.values >= 1) & (raw.fold.values >= 0))[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values


def cross_feats(df, spec):
    d = df.sort_values(["experiment_id", "relative_time_s"]); eid, t = d.experiment_id, d.relative_time_s; g = d.groupby("experiment_id"); out = {}
    for name, (fn, levels) in spec.items():
        cm = fn(d, g).groupby(eid).cummax()
        for lv in levels: out[f"cx_{name}_{lv}"] = t - t.where(cm >= lv).groupby(eid).cummin()
    return pd.DataFrame(out).loc[df.index]


SPEC1 = {"thermal": (lambda d, g: d.thermal_t, [35, 40, 45, 50, 55, 60, 65, 70, 80, 100, 150, 200]),
         "thermal_rel": (lambda d, g: d.thermal_t - g.thermal_t.transform("first"), [2, 5, 10, 15, 20, 30, 40])}
base = F.build_features(raw)
X = pd.concat([base, cross_feats(raw, SPEC1).set_index(base.index)], axis=1).iloc[lab_i].reset_index(drop=True)
X.to_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet")
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
          reg_lambda=1.0, class_weight="balanced", random_state=42, n_jobs=16, verbose=-1)
CFG = {"C0 기준": {}, "C1 leaves15": dict(num_leaves=15), "C2 leaves63": dict(num_leaves=63), "C3 colsample0.15": dict(colsample_bytree=0.15),
       "C4 colsample0.5": dict(colsample_bytree=0.5), "C5 min_child20": dict(min_child_samples=20), "C6 min_child150": dict(min_child_samples=150),
       "C7 500트리 lr0.03": dict(n_estimators=500, learning_rate=0.03), "C8 lambda10": dict(reg_lambda=10.0), "C9 extra_trees": dict(extra_trees=True),
       "C10 no_balanced": dict(class_weight=None)}
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
base_fs = None
for nm, ch in CFG.items():
    t0 = time.time(); O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k
        O[va] = lgb.LGBMClassifier(**{**P0, **ch}).fit(X[~va], y[~va]).predict_proba(X[va])
    p = F.postprocess(fr, O, 0.7)
    fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    if base_fs is None: base_fs = fs
    np.save(ROOT / "outputs" / "preds" / f"tune_{nm.split()[0]}_oof.npy", O.astype(np.float32))
    print(f"{nm:20s} OOF {f1(y, p):.4f} ({f1(y, p) - f1(y, F.postprocess(fr, np.load(ROOT / 'outputs/preds/tune_C0_oof.npy'), 0.7)):+.4f}) 이긴 fold {int((fs > base_fs).sum())}/5 | fold {np.round(fs, 4).tolist()} | {time.time() - t0:.0f}s", flush=True)
