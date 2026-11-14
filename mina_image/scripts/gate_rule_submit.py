"""즉시 제출용: 센서(val0.9195, Train-only 모델) + 이미지(ft_ord_dinov2_b) 게이트 규칙으로 제출 CSV 생성.
규칙: 센서 후처리 예측 == L2 이고, 센서 확신도 < c 이고, 이미지 P(>=L3) > theta 이면 L3 로 올린다. 이후 단조 보정(cummax, tau=0.5)을 한 번 더 적용.
주의: c/theta 는 Validation 오류를 보고 훑어서 골랐다(탐색 결과, 낙관적). 센서 OOF 기반 정식 평가는 run_fusion_plan.sh 가 따로 한다.
이미지 영향도: 이미지 확률을 중립값(학습 OOF 평균)으로 바꿨을 때 바뀌는 행 수(빈 이미지 대용 근사)."""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F

L = list(range(1, 7))
mj = joblib.load("/home/ubuntu/nicole/val0.9195/model.joblib")
mod = mj["model"]; mod.set_params(n_jobs=8)
rd = lambda s: pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}).assign(sample_id=lambda d: d.sample_id.astype(str))
val, tst = rd("validation"), rd("test")
vcap = F.cap_rows(val).reset_index(drop=True)
sp = lambda fr: mod.predict_proba(F.build_features(fr)[mj["features"]])
Pv, Pvc, Pt = sp(val), sp(vcap), sp(tst)
t = table(); lab = (t.label.values >= 1) & (t.fold.values >= 0); te = t.label.values == -1
oof = np.load(ROOT / "outputs/preds/ft_ord_dinov2_b_oof.npy"); tes = np.load(ROOT / "outputs/preds/ft_ord_dinov2_b_test.npy")
Iv = pd.DataFrame(oof, index=t.sample_id[lab].astype(str).values); It = pd.DataFrame(tes, index=t.sample_id[te].astype(str).values)
prior = oof.mean(0, keepdims=True)
img = lambda fr, src: src.loc[fr.sample_id.values].values
hard = lambda fr, p: F.postprocess(fr, np.eye(6)[p - 1], 0.5)


def gate(fr, Ps, Pi, c, th):
    ps = F.postprocess(fr, Ps, 0.7)
    up = (ps == 2) & (Ps.max(1) < c) & (Pi[:, 2:].sum(1) > th)
    p = ps.copy(); p[up] = 3
    return hard(fr, p), hard(fr, ps), int(up.sum())


f1 = lambda y, p: f1_score(y, p, labels=L, average="macro", zero_division=0)
order = pd.read_csv(DATA / "sample_submission.csv", dtype={"sample_id": "string"})["sample_id"].astype(str).tolist()
ref = pd.read_csv(ROOT / "outputs/submissions/ens_lgb_xgb.csv", dtype={"sample_id": "string"}).set_index("sample_id").label
for name, c, th in [("gate_c09_t08", 0.9, 0.8), ("gate_c099_t09", 0.99, 0.9)]:
    print(f"\n=== {name}: 센서 확신도<{c}, 이미지 P(>=L3)>{th}")
    for nm, fr, Ps in [("Val cap500", vcap, Pvc), ("Val full", val, Pv)]:
        g, b, n = gate(fr, Ps, img(fr, Iv), c, th)
        gn, _, _ = gate(fr, Ps, np.tile(prior, (len(fr), 1)), c, th)
        y = fr.label.astype(int).values
        print(f"  {nm:10s}: 기준(센서+단조) {f1(y, b):.4f} -> 게이트 {f1(y, g):.4f} ({f1(y, g) - f1(y, b):+.4f}) | 올린 행 {n} | 이미지 중립화 시 바뀌는 행 {int((g != gn).sum())}")
    g, b, n = gate(tst, Pt, img(tst, It), c, th)
    gn, _, _ = gate(tst, Pt, np.tile(prior, (len(tst), 1)), c, th)
    s = pd.Series(g.astype(int), index=tst.sample_id.values)
    assert set(s.index) == set(order)
    sub = pd.DataFrame({"sample_id": order, "label": s.loc[order].values})
    assert len(sub) == 3451 and sub.label.isin(L).all() and sub.sample_id.is_unique
    out = ROOT / "outputs/submissions" / f"{name}.csv"
    sub.to_csv(out, index=False)
    print(f"  TEST: 올린 행 {n}/3451, 이미지 중립화 시 바뀌는 행 {int((g != gn).sum())} | 센서단독(후처리)과 다른 행 {int((g != b).sum())}")
    print("  라벨 분포:", sub.label.value_counts(normalize=True).sort_index().round(3).to_dict(), "| ens_lgb_xgb.csv 와 일치율", round(float((ref.loc[sub.sample_id.values].values == sub.label.values).mean()), 4))
    print("  저장:", out)
