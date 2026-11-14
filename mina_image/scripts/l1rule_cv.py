"""eda_v2 의 근거를 반영한 두 가지 학습 설계를 OOF 5-fold(fast, X1 피처 캐시)로 검증.
 L1 규칙: L1->L2 는 'T1 누적최대 >= 45°C' 규칙이 97/97 정확 -> 규칙으로 L1 을 고정하고, 규칙 통과(cummax>=45) 프레임만으로 L2~L6 모델을 학습·예측.
 cap500 가중: Test 는 9개 실험이 원본(최대 325행), 2개가 500행 -> 학습에서 실험당 가중 w=min(1, 500/n) 로 긴 실험 지배를 줄임.
변형: base / l1rule / cap500 / l1rule+cap500. 지표: 합산 OOF F1, 실험당 500행 상한 가중 F1(Test 유사), bat_class별."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=["sample_id", "experiment_id", "meas_t1", "relative_time_s", "bat_class"]) for s in ["train", "validation"]])
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index()          # fr 행 순서
order = np.lexsort((raw.relative_time_s.values, raw.experiment_id.values))
cm = np.empty(len(raw)); cm[order] = raw.iloc[order].groupby("experiment_id").meas_t1.cummax().values   # 실험별 T1 누적최대(과거만)
passed = cm >= 45
print("규칙 정확도: (규칙통과 == 라벨>=2) 프레임 비율", round(float((passed == (y >= 2)).mean()), 5), flush=True)
cls = raw.bat_class.values
X = pd.read_parquet(ROOT / "outputs" / "preds" / "X1_lab.parquet")
P0 = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
          class_weight="balanced", n_jobs=16, verbose=-1, random_state=42)
n = fr.groupby("experiment_id").experiment_id.transform("size").values.astype(float); wcap = np.minimum(1.0, 500.0 / n)
f1 = lambda yy, p, w=None: f1_score(yy, p, labels=L, average="macro", zero_division=0, sample_weight=w)
base_f = None
for nm, l1, wc in [("base", False, False), ("l1rule", True, False), ("cap500", False, True), ("l1rule+cap500", True, True)]:
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k
        trm = (~va) & (passed if l1 else np.ones(len(y), bool))
        w = wcap if wc else np.ones(len(y))
        m = lgb.LGBMClassifier(**P0).fit(X[trm], y[trm], sample_weight=w[trm])
        pr = m.predict_proba(X[va]); P = np.zeros((va.sum(), 6)); P[:, np.array(m.classes_) - 1] = pr
        if l1:  # 규칙 미통과 -> L1 확정, 통과 -> 모델 예측(L2~L6)
            pa = passed[va]; P[~pa] = 0; P[~pa, 0] = 1.0; P[pa, 0] = 0
        O[va] = P
    p = F.postprocess(fr, O, 0.7)
    fs = np.array([f1(y[fold == k], p[fold == k]) for k in range(5)])
    if base_f is None: base_f, base_fs = f1(y, p), fs
    print(f"{nm:16s} 합산 OOF {f1(y, p):.4f} ({f1(y, p) - base_f:+.4f}) | 500행 상한 가중 F1 {f1(y, p, wcap):.4f} | 이긴 fold {int((fs > base_fs).sum())}/5 | bat_class별 {[round(f1(y[cls == c], p[cls == c]), 4) for c in (1, 2, 3)]} | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True)
