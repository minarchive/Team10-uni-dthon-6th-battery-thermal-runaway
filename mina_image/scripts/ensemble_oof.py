"""저장된 OOF 확률로 (1) 모델 간 오류 겹침, (2) 확률 평균 앙상블, (3) 실험 단위 부트스트랩 신뢰구간을 계산.

대상: best_layer_by_encoder.csv 의 인코더별 최고 레이어(frozen probe) + outputs/preds/ft_*_oof.npy (파인튜닝)
핵심 해석: 'all_wrong' 비율이 높으면 한 프레임만으로는 구분 불가(정보 한계) -> 과거 프레임/센서가 필요.
산출: outputs/logs/ensemble_report.txt (stdout 과 동일)
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import ROOT, table

P = ROOT / "outputs" / "preds"
df = table()
lab = (df.label.values >= 1) & (df.fold.values >= 0)
y = df.label.values[lab] - 1
grp = df.experiment_id.values[lab]
L = list(range(6))


def mf1(yt, yp):
    return f1_score(yt, yp, average="macro", labels=L, zero_division=0)


models = {}
best = pd.read_csv(ROOT / "outputs" / "logs" / "best_layer_by_encoder.csv")
for _, r in best.iterrows():
    if "distort" in r.encoder or "smoke" in r.encoder:
        continue
    f = P / f"{r.encoder}__{r.layer}_oof.npy"
    if f.exists():
        models[f"{r.encoder}:{r.layer}"] = np.load(f)
for f in sorted(P.glob("ft_*_oof.npy")):
    models["FT_" + f.stem[3:-4]] = np.load(f)
names = list(models)
pred = {k: v.argmax(1) for k, v in models.items()}
wrong = {k: pred[k] != y for k in names}

print("== 개별 모델 (OOF Macro F1)")
for k in names:
    print(f"{k:32s} F1 {mf1(y, pred[k]):.4f}  acc {1 - wrong[k].mean():.4f}")

W = np.stack([wrong[k] for k in names])
print(f"\n== 오류 겹침 (프레임 {len(y)}개, 모델 {len(names)}개)")
print(f"한 모델이라도 틀린 비율 {W.any(0).mean():.4f} | 모두 틀린 비율 {W.all(0).mean():.4f} | 평균 개별 오류율 {W.mean():.4f}")
print("-> 모두 틀린 비율 / 평균 개별 오류율 = %.2f (1에 가까울수록 같은 프레임에서 같이 틀림)" % (W.all(0).mean() / W.mean()))
jac = pd.DataFrame(index=names, columns=names, dtype=float)
for a, b in itertools.product(names, names):
    jac.loc[a, b] = (wrong[a] & wrong[b]).sum() / max(1, (wrong[a] | wrong[b]).sum())
print("\n오류 Jaccard (두 모델의 오류 집합 겹침):")
print(jac.round(2).to_string())

print("\n== 확률 평균 앙상블")
ens_all = np.mean([models[k] for k in names], axis=0)
single_best = max(names, key=lambda k: mf1(y, pred[k]))
print(f"전체 평균 F1 {mf1(y, ens_all.argmax(1)):.4f} (최고 단일 {single_best} {mf1(y, pred[single_best]):.4f})")
fam = {}
for k in names:
    fam.setdefault(k.split(":")[0].replace("FT_", ""), k)
for n in (2, 3):
    for combo in itertools.combinations(list(fam.values()), n):
        pass
top = sorted(names, key=lambda k: -mf1(y, pred[k]))[:3]
e3 = np.mean([models[k] for k in top], axis=0)
print(f"상위 3개 평균 {top}: F1 {mf1(y, e3.argmax(1)):.4f}")

print("\n== 실험 단위 부트스트랩 (전체 평균 앙상블 - 최고 단일), 200회")
rng = np.random.default_rng(0)
ug = np.unique(grp)
idx = {g: np.where(grp == g)[0] for g in ug}
diffs = []
pe, ps = ens_all.argmax(1), pred[single_best]
for _ in range(200):
    s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
    diffs.append(mf1(y[s], pe[s]) - mf1(y[s], ps[s]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"차이 평균 {np.mean(diffs):+.4f}, 95% 구간 [{lo:+.4f}, {hi:+.4f}] -> {'0 포함: 유의하지 않음' if lo <= 0 <= hi else '0 미포함: 유의한 차이'}")

print("\n== 단계별 F1 (전체 평균 앙상블)")
print([round(v, 3) for v in f1_score(y, pe, average=None, labels=L, zero_division=0)])
