"""어려운 실험 겹침 + 실험 단위 부트스트랩(Test 모사: 실험당 500행 상한 가중).
질문: (1) LightGBM 과 신경망이 같은 실험에서 같이 틀리는가(오류 상관) (2) 모델 간 차이가 실험 단위 부트스트랩에서 유의한가 (3) fold 단위 변동이 Test 11개 실험 변동과 비슷한가.
사용: python scripts/hard_exp.py final_lgb_oof ft_mm_all_hdtr_oof_gated ... (확률 npy 는 후처리(tau .7) 후, 1차원이면 단계 예측으로 간주)"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7))
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str)
y = fr.label.astype(int).values; fold = fr.fold.values; ex = fr.experiment_id.values
P = ROOT / "outputs" / "preds"
f1 = lambda a, b, w=None: f1_score(a, b, labels=L, average="macro", zero_division=0, sample_weight=w)
pred = {}
for n in sys.argv[1:]:
    p = np.load(P / f"{n}.npy"); pred[n] = p if p.ndim == 1 else F.postprocess(fr, p, 0.7)
ue = np.unique(ex); eidx = {e: np.where(ex == e)[0] for e in ue}
cnt = pd.Series(ex).map(pd.Series(ex).value_counts()).values.astype(float); wcap = np.minimum(1.0, 500.0 / cnt)
info = fr.groupby("experiment_id").agg(bc=("bat_class", "first"), tr=("tr_method", "first"), soc=("soc_status", "first"), cc=("bat_cc", "first"), n=("label", "size"), fold=("fold", "first"))
for n, p in pred.items(): info["acc_" + n[:14]] = pd.Series((p == y).astype(float)).groupby(ex).mean().reindex(info.index).values
cols = [c for c in info.columns if c.startswith("acc_")]
print("모델별 합산 F1:", {n[:20]: round(f1(y, p), 4) for n, p in pred.items()}, "| 500행 상한 가중:", {n[:20]: round(f1(y, p, wcap), 4) for n, p in pred.items()})
print("\n실험별 정확도 상관(오류 겹침):\n", info[cols].corr().round(2).to_string())
for c in cols:
    w = info.sort_values(c).head(10)
    print(f"\n[{c}] 최하위 10개 실험:\n", w[["bc", "tr", "soc", "cc", "n", "fold", c]].round(3).to_string())
if len(cols) >= 2:
    a, b = cols[0], cols[1]
    wa = set(info.sort_values(a).head(10).index); wb = set(info.sort_values(b).head(10).index)
    print(f"\n최하위 10개 교집합 {len(wa & wb)}개: {sorted(wa & wb)}")
rng = np.random.default_rng(0); names = list(pred)
B = 400; res = {n: [] for n in names}
for _ in range(B):
    s = rng.choice(ue, len(ue), replace=True); ix = np.concatenate([eidx[e] for e in s]); w = wcap[ix]
    for n in names: res[n].append(f1(y[ix], pred[n][ix], w))
res = {n: np.array(v) for n, v in res.items()}
print("\n실험 단위 부트스트랩(500행 가중 F1): 평균 ± 표준편차(5~95%)")
for n in names: print(f"  {n[:26]:26s} {res[n].mean():.4f} ± {res[n].std():.4f} ({np.percentile(res[n], 5):.3f}~{np.percentile(res[n], 95):.3f})")
for n in names[1:]:
    d = res[n] - res[names[0]]; print(f"  {n[:26]:26s} - {names[0][:12]}: {d.mean():+.4f}  [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] 개선 비율 {(d > 0).mean():.2f}")
# fold 점수 분산 vs 무작위 19개 실험 부분집합의 분산
fs = np.array([f1(y[fold == k], pred[names[0]][fold == k]) for k in range(5)])
sub = []
for _ in range(400):
    s = rng.choice(ue, 19, replace=False); ix = np.concatenate([eidx[e] for e in s]); sub.append(f1(y[ix], pred[names[0]][ix]))
sub11 = []
for _ in range(400):
    s = rng.choice(ue, 11, replace=False); ix = np.concatenate([eidx[e][:500] for e in s]); sub11.append(f1(y[ix], pred[names[0]][ix]))
print(f"\n{names[0][:20]}: 실제 fold 5개 F1 표준편차 {fs.std(ddof=1):.4f} | 무작위 19개 실험 부분집합 표준편차 {np.std(sub):.4f} | 무작위 11개(500행 상한) 표준편차 {np.std(sub11):.4f}")
