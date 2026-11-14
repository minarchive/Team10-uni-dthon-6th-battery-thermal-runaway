"""종단간(센서 확률 입력 + 이미지 백본 파인튜닝) OOF 를 센서 단독과 비교. 같은 fold, 같은 후처리."""
import sys
import numpy as np
from sklearn.metrics import f1_score
from common import ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
name = sys.argv[1] if len(sys.argv) > 1 else "e2e_sens"
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
so = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy"); eo = np.load(ROOT / f"outputs/preds/ft_{name}_oof.npy")
f = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
ps, pe = F.postprocess(fr, so, 0.7), F.postprocess(fr, eo, 0.7)
fs = np.array([f(y[fold == k], ps[fold == k]) for k in range(5)]); fe = np.array([f(y[fold == k], pe[fold == k]) for k in range(5)])
trm = fr.tr_method.values
print(f"센서 OOF(후처리) {f(y, ps):.4f} | 종단간({name}) {f(y, pe):.4f} ({f(y, pe) - f(y, ps):+.4f}) | 이긴 fold {int((fe > fs).sum())}/5 | fold차이 {np.round(fe - fs, 4).tolist()}")
print(f"  가열 {f(y[trm==1], ps[trm==1]):.4f} -> {f(y[trm==1], pe[trm==1]):.4f} | 과충전 {f(y[trm==2], ps[trm==2]):.4f} -> {f(y[trm==2], pe[trm==2]):.4f}")
print("  단계별 센서", np.round(f1_score(y, ps, labels=L, average=None), 3).tolist(), "-> 종단간", np.round(f1_score(y, pe, labels=L, average=None), 3).tolist())
ch = pe != ps; print(f"  바뀐 프레임 {int(ch.sum())} 중 정답으로 {int((ch & (pe == y)).sum())}, 오답으로 {int((ch & (ps == y)).sum())}")
for w in (0.3, 0.5, 0.7):
    pb = F.postprocess(fr, (1 - w) * so + w * eo, 0.7); print(f"  블렌드 w={w}: {f(y, pb):.4f} ({f(y, pb) - f(y, ps):+.4f})")
