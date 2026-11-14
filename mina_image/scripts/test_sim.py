"""예상 Test 점수 추정(팀 방식 적용): (1) Test 의 그룹(형태x방식) 행 비중으로 OOF 재가중한 Macro-F1 (2) Test 와 같은 구성(그룹별 실험 수·실험별 행 수, 500행 상한은 균등 간격 추출)으로 같은 그룹에서 학습 실험을 무작위로 뽑은 가상 Test 1500회의 점수 분포.
입력: OOF 확률 .npy (라벨 행 순서). 후처리: 확신 cummax tau=0.7 (가상 Test 에서는 추출한 행에서 다시 적용). 신경망 OOF 는 --gate 로 L1 규칙 후처리 적용."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values
cols = ["experiment_id", "bat_class", "tr_method", "relative_time_s", "meas_t1"]
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=cols + (["sample_id"])) for s in ["train", "validation"]]); raw["sample_id"] = raw.sample_id.astype(str); raw = raw.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index(); raw["experiment_id"] = raw.experiment_id.astype(str)
te = pd.read_csv(DATA / "test" / "test.csv", usecols=["experiment_id", "bat_class", "tr_method"]); te["experiment_id"] = te.experiment_id.astype(str)
grp = lambda d: (d.bat_class.astype(int) * 10 + d.tr_method.astype(int))
fr["grp"] = grp(raw).values; te["grp"] = grp(te)
tstruct = te.groupby("experiment_id").agg(grp=("grp", "first"), n=("grp", "size")).reset_index()
eid = raw.experiment_id.values; t = raw.relative_time_s.values.astype(float); t1 = raw.meas_t1.values.astype(float); cm = np.zeros(len(y))
for e, ix in pd.Series(np.arange(len(y))).groupby(eid).indices.items():
    ix = ix[np.argsort(t[ix], kind="stable")]; cm[ix] = np.maximum.accumulate(t1[ix])
reached = cm >= 45
def gate(P):
    G = P.copy(); G[~reached] = 0; G[~reached, 0] = 1.0; G[reached, 0] = 0; s = G[reached].sum(1, keepdims=True); G[reached] = np.where(s > 0, G[reached] / np.where(s > 0, s, 1), 0.2); G[reached, 0] = 0; return G
trg = fr.groupby("grp").size() / len(fr); teg = te.groupby("grp").size() / len(te)
wmap = (teg / trg).to_dict(); w = fr.grp.map(wmap).fillna(0).values
idx_by_exp = {e: np.where(eid == e)[0][np.argsort(t[eid == e], kind="stable")] for e in np.unique(eid)}
exps_by_grp = {g: [e for e in idx_by_exp if fr.grp.values[idx_by_exp[e][0]] == g] for g in fr.grp.unique()}
rng = np.random.default_rng(0)
def simulate(P, n_rep=1000):
    sc = []
    for _ in range(n_rep):
        ys, ps = [], []
        for _, r in tstruct.iterrows():
            e = exps_by_grp[r.grp][rng.integers(len(exps_by_grp[r.grp]))]; ix = idx_by_exp[e]
            if len(ix) > r.n: ix = ix[np.unique(np.round(np.linspace(0, len(ix) - 1, r.n)).astype(int))]
            sub = fr.iloc[ix].reset_index(drop=True); ys.append(y[ix]); ps.append(F.postprocess(sub, P[ix], 0.7))
        sc.append(f1_score(np.concatenate(ys), np.concatenate(ps), labels=L, average="macro", zero_division=0))
    return np.array(sc)
print("Test 그룹 구성(실험/행 비중):", {int(g): (int((tstruct.grp == g).sum()), round(float(v), 3)) for g, v in teg.items()})
for spec in sys.argv[1:]:
    name, _, gt = spec.partition(":"); P = np.load(ROOT / "outputs/preds" / f"{name}.npy")
    if gt == "gate": P = gate(P)
    p = F.postprocess(fr, P, 0.7); pooled = f1_score(y, p, labels=L, average="macro", zero_division=0)
    rew = f1_score(y, p, labels=L, average="macro", sample_weight=w, zero_division=0)
    sc = simulate(P)
    print(f"{spec:34s} 풀링 OOF {pooled:.4f} | Test 구성 재가중 {rew:.4f} | 가상 Test 평균 {sc.mean():.4f} ± {sc.std():.4f} (5~95% {np.percentile(sc, 5):.3f}~{np.percentile(sc, 95):.3f})", flush=True)
