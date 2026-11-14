"""indu_t 유무(실험 단위 전부 결측 vs 있음)에 따른 센서 모델 성능 차이와 처리 방식 비교 (OOF 5-fold, fast 설정).
변형: base(현재) / no_indu(indu_t 신호 제외) / split(유무별 별도 모델) / split_noindu(없는 실험용 모델은 indu 제외, 있는 쪽은 유지)."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F
L = list(range(1, 7)); BASE0 = list(F.BASE_SIGNALS)
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
has = raw.groupby("experiment_id").indu_t.transform(lambda s: s.notna().any()).values
lab_i = np.where((raw.label.values >= 1) & (raw.fold.values >= 0))[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y = fr.label.astype(int).values; fold = fr.fold.values; hv = has[lab_i]


def feats(frame, drop_indu):
    F.BASE_SIGNALS = [s for s in BASE0 if not (drop_indu and s == "indu_t")]
    F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    return F.build_features(frame.reset_index(drop=True))


def mk():
    m = F.build_model(); m.set_params(n_jobs=16, n_estimators=250, learning_rate=0.06); return m


f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
X1, X0 = feats(raw, False).iloc[lab_i].reset_index(drop=True), feats(raw, True).iloc[lab_i].reset_index(drop=True)
out = {}
def run(name, fn):
    O = np.zeros((len(y), 6))
    for k in range(5):
        va = fold == k; O[va] = fn(va)
    p = F.postprocess(fr, O, 0.7)
    print(f"{name:14s} OOF {f1(y, p):.4f} | indu 있음 {f1(y[hv], p[hv]):.4f} | indu 없음 {f1(y[~hv], p[~hv]):.4f} | fold {[round(f1(y[fold==k], p[fold==k]),4) for k in range(5)]}", flush=True)
    out[name] = O
run("base", lambda va: mk().fit(X1[~va], y[~va]).predict_proba(X1[va]))
run("no_indu", lambda va: mk().fit(X0[~va], y[~va]).predict_proba(X0[va]))
def split(XA, XB):
    def fn(va):
        P = np.zeros((va.sum(), 6)); hvv = hv[va]
        for flag, X in [(True, XA), (False, XB)]:
            tr = (~va) & (hv == flag)
            te = hvv == flag
            if te.any(): P[te] = mk().fit(X[tr], y[tr]).predict_proba(X[va][te])
        return P
    return fn
run("split", split(X1, X1))
run("split_noindu", split(X1, X0))
