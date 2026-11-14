"""클래스별 결정 편향(로짓 바이어스)과 클래스별 후처리 임계값을 Macro-F1 에 맞게 최적화. 중첩 검증: 4개 fold 의 OOF 확률로 편향을 정하고 나머지 1개 fold 에 적용.
확률 P(n,6) -> log P + b -> softmax -> 팀 후처리(확신 cummax). b 는 좌표 상승으로 OOF Macro-F1 을 최대화(격자 -1.0~1.0, 0.25 간격, 2 라운드)."""
import sys, time
import numpy as np
from sklearn.metrics import f1_score
from common import ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
P = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy")
lg = np.log(np.clip(P, 1e-6, 1))
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
def apply(b, rows, tau=0.7):
    z = lg[rows] + b; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return F.postprocess(fr.iloc[rows].reset_index(drop=True), p, tau)
def fit(rows, tau=0.7, grid=np.arange(-1.0, 1.01, 0.25), rounds=2):
    b = np.zeros(6); best = f1(y[rows], apply(b, rows, tau))
    for _ in range(rounds):
        for c in range(1, 6):            # L1 편향은 고정(규칙으로 정확)
            for g in grid:
                bb = b.copy(); bb[c] = g; s = f1(y[rows], apply(bb, rows, tau))
                if s > best + 1e-6: best, b = s, bb
    return b, best
base = np.array([f1(y[fold == k], apply(np.zeros(6), np.where(fold == k)[0])) for k in range(5)])
print(f"기준 OOF(후처리 τ=0.7) {f1(y, apply(np.zeros(6), np.arange(len(y)))):.4f} fold {np.round(base, 4).tolist()}", flush=True)
res = []; allb = []
for k in range(5):
    t0 = time.time(); tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
    b, s_tr = fit(tr); s_te = f1(y[te], apply(b, te)); res.append(s_te); allb.append(b)
    print(f"fold {k}: 편향 {np.round(b, 2).tolist()} | 학습 fold F1 {s_tr:.4f} -> 보지 않은 fold {s_te:.4f} (기준 {base[k]:.4f}, {s_te - base[k]:+.4f}) | {time.time() - t0:.0f}s", flush=True)
res = np.array(res); print(f"\n[중첩] 평균 {res.mean():.4f} (기준 {base.mean():.4f}, {res.mean() - base.mean():+.4f}) 이긴 fold {int((res > base).sum())}/5")
bm = np.mean(allb, 0); print("fold 평균 편향:", np.round(bm, 2).tolist(), "| 전체 OOF 에 적용(낙관적):", round(f1(y, apply(bm, np.arange(len(y)))), 4))
