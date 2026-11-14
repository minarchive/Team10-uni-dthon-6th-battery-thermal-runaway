"""오류 분해 기반 규칙 확장(최종 LightGBM OOF 위, 모두 인과적·학습 fold 에서만 임계 결정).
R1 가열 L3 규칙: 가열 행에서 모델이 L2/L3 로 본 프레임을 T1 상승량(첫 관측 대비) 누적최댓값 >= theta_c(형태별, 학습 fold 가열 실험의 L2/L3 프레임에서 F1 최적)로 재결정
R2 L3 최소 시각: T1 45°C 도달 후 경과 시간이 (형태x방식) 학습 실험의 'L2 지속 시간' q 분위수보다 짧으면 L3 예측을 L2 로(너무 이른 L3 방지)
R3 L3 최대 지연 방지는 사후 정보가 필요해 제외."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=["sample_id", "experiment_id", "relative_time_s", "meas_t1", "bat_class", "tr_method"]) for s in ["train", "validation"]])
raw["sample_id"] = raw.sample_id.astype(str); raw = raw.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index(); raw["experiment_id"] = raw.experiment_id.astype(str)
eid = raw.experiment_id.values; t = raw.relative_time_s.values.astype(float); t1 = raw.meas_t1.values.astype(float); cls = raw.bat_class.values.astype(int); tr = raw.tr_method.values.astype(int)
idx = pd.Series(np.arange(len(y))).groupby(eid).indices
t1rel_cm = np.zeros(len(y)); since45 = np.full(len(y), np.nan)
for e, ix in idx.items():
    ix = ix[np.argsort(t[ix], kind="stable")]; t1rel_cm[ix] = np.maximum.accumulate(t1[ix] - t1[ix][0]); cm = np.maximum.accumulate(t1[ix]); m = cm >= 45
    if m.any(): tf = t[ix][np.argmax(m)]; since45[ix] = np.where(t[ix] >= tf, t[ix] - tf, np.nan)
f1 = lambda p: f1_score(y, p, labels=L, average="macro", zero_division=0)
Pb = np.load(ROOT / "outputs/preds/final_lgb_oof.npy"); pb = F.postprocess(fr, Pb, 0.7); base = f1(pb)
fb = np.array([f1_score(y[fold == k], pb[fold == k], labels=L, average="macro") for k in range(5)])
print(f"기준(final_lgb OOF, 과충전 규칙 포함) {base:.4f} | fold {np.round(fb, 3).tolist()}")
def rep(nm, P):
    p = F.postprocess(fr, P, 0.7); fs = np.array([f1_score(y[fold == k], p[fold == k], labels=L, average="macro") for k in range(5)]); fv = f1(p)
    print(f"{nm:44s} OOF {fv:.4f} ({fv - base:+.4f}) 이긴 fold {int((fs > fb).sum())}/5 | 단계별 {np.round(f1_score(y, p, labels=L, average=None), 3).tolist()}", flush=True); return fv
def override(P, rows, l3):
    P = P.copy(); p23 = P[rows, 1] + P[rows, 2]; sel = (P[rows].argmax(1) >= 1) & (P[rows].argmax(1) <= 2); r = rows[sel]
    P[r, 1] = np.where(l3[sel], 0.0, p23[sel]); P[r, 2] = np.where(l3[sel], p23[sel], 0.0); return P
# R1 가열 L3 규칙 (형태별 theta, 학습 fold 에서)
for scope in ("형태별 theta", "가열 전체 theta"):
    P = Pb.copy()
    for k in range(5):
        va = fold == k
        for c in (1, 2, 3):
            trm = (~va) & (tr == 1) & ((cls == c) if scope == "형태별 theta" else True) & np.isin(y, [2, 3]); v = t1rel_cm[trm]
            grid = np.unique(np.percentile(v, np.linspace(2, 98, 33))); b, th = -1, grid[0]
            for g in grid:
                f = f1_score((y[trm] == 3).astype(int), (v >= g).astype(int), zero_division=0)
                if f > b: b, th = f, g
            rows = np.where(va & (tr == 1) & (cls == c))[0]
            if len(rows): P = override(P, rows, t1rel_cm[rows] >= th)
    rep(f"R1 가열 L3 규칙 ({scope})", P)
# R2 L3 최소 시각
l2dur = {}
for e, ix in idx.items():
    ix = ix[np.argsort(t[ix], kind="stable")]; yy = y[ix]; tt = t[ix]
    if (yy >= 2).any() and (yy >= 3).any(): l2dur[e] = (tt[yy >= 3].min() - tt[yy >= 2].min(), int(cls[ix][0]), int(tr[ix][0]), int(fold[ix][0]))
for q in (5, 10, 20):
    P = Pb.copy()
    for k in range(5):
        for c in (1, 2, 3):
            for m in (1, 2):
                ds = [d for d, cc, mm, ff in l2dur.values() if ff != k and cc == c and mm == m]
                if len(ds) < 3: continue
                thr = np.percentile(ds, q); rows = np.where((fold == k) & (cls == c) & (tr == m) & (~np.isnan(since45)) & (since45 < thr))[0]
                pm = P[rows].argmax(1) == 2; r = rows[pm]; P[r, 1] += P[r, 2]; P[r, 2] = 0.0
    rep(f"R2 L3 최소 시각 (L2 지속 {q}% 분위수 이전 L3 금지)", P)
print("\n(참고) L2 지속 시간 분위수 (그룹, 초): ", {f"{['','원통','파우치','각형'][c]}-{['','가열','과충전'][m]}": [int(np.percentile([d for d, cc, mm, ff in l2dur.values() if cc == c and mm == m], q)) for q in (5, 10, 50)] for c in (1, 2, 3) for m in (1, 2) if any(cc == c and mm == m for d, cc, mm, ff in l2dur.values())})
