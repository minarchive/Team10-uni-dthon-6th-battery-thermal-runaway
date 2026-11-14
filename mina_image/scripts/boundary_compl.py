"""경계(단계 j -> j+1)별로 센서와 이미지가 서로 다른 정보를 주는가. 전환 시점 ±W초 프레임만 사용, 실험 단위 fold CV.
점수: 센서 OOF P(>=j+1), 이미지(순서형 ft_ord_dinov2_b) OOF P(>=j+1). 지표: AUC(센서/이미지/결합 로지스틱), 전환 시점 검출 오차(점수가 처음 0.5 이상이 된 시각 - 실제 전환 시각)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import ROOT, table
W = 300
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str)
y = fr.label.astype(int).values; fold = fr.fold.values; t = fr.relative_time_s.values.astype(float); eid = fr.experiment_id.values
so = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy"); io = np.load(ROOT / "outputs/preds/ft_ord_dinov2_b_oof.npy")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
print(f"전환 ±{W}초 프레임, 실험 단위 5-fold. AUC: 센서 / 이미지 / 결합(로지스틱) | 전환 시점 검출 오차 중앙값(초, 음수=일찍): 센서 / 이미지 / 결합")
for j in range(1, 6):
    s = so[:, j:].sum(1); q = io[:, j:].sum(1)
    onset = {}
    for e in np.unique(eid):
        m = (eid == e) & (y >= j + 1)
        if m.any(): onset[e] = t[m].min()
    near = np.zeros(len(y), bool)
    for e, to in onset.items(): near |= (eid == e) & (np.abs(t - to) <= W)
    tgt = (y >= j + 1).astype(int)
    if tgt[near].min() == tgt[near].max(): continue
    X = np.stack([lg(s), lg(q)], 1); comb = np.zeros(len(y))
    for k in range(5):
        tr, va = near & (fold != k), near & (fold == k)
        if tgt[tr].min() == tgt[tr].max() or va.sum() == 0: continue
        comb[va] = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], tgt[tr]).predict_proba(X[va])[:, 1]
    a_s, a_i = roc_auc_score(tgt[near], s[near]), roc_auc_score(tgt[near], q[near]); a_c = roc_auc_score(tgt[near], comb[near])
    def timing(score):
        errs = []
        for e, to in onset.items():
            m = np.where((eid == e))[0]; m = m[np.argsort(t[m])]
            hit = m[score[m] >= 0.5]
            errs.append((t[hit[0]] - to) if len(hit) else np.nan)
        errs = np.array(errs); return np.nanmedian(errs), np.nanmedian(np.abs(errs)), np.isnan(errs).mean()
    ts, ti, tc = timing(s), timing(q), timing(comb)
    print(f"L{j}->L{j+1}: 실험 {len(onset)} 프레임 {near.sum()} | AUC {a_s:.3f} / {a_i:.3f} / {a_c:.3f} (결합-센서 {a_c - a_s:+.3f}) | 오차 중앙 {ts[0]:+.0f} / {ti[0]:+.0f} / {tc[0]:+.0f}, 절대 중앙 {ts[1]:.0f} / {ti[1]:.0f} / {tc[1]:.0f}")
