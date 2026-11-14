"""CPU 분석 4종(트리 모델 새 학습 없음, 기존 OOF 재사용): 1 과충전/가열 L2 vs L3 저용량 모델+중첩 피처 선택, 2 가스 상대화(로지스틱), 3 SOC 100% 진행 속도, 4 단계 시작 시각 오차.
인과성: 모델 입력 피처는 같은 실험의 현재·과거 관측(누적 극값, 최근 30초 기울기)만 사용. 실험 3·4 의 실험 전체 통계는 분석용(입력 아님).
평가: 실험 단위 5-fold 를 시드 3개로 무작위 배정(group), 비교는 실험별 정확도 차이의 부트스트랩 95% CI(2000회). 결과: outputs/DA_REPORT.md (+stdout).
사용: python scripts/da_cpu.py [--only 1,3,4,2] [--n_jobs 2]"""
import argparse, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--only", default="1,3,4,2"); ap.add_argument("--n_jobs", type=int, default=2); ap.add_argument("--oof", default="final_lgb_oof"); a = ap.parse_args()
T0 = time.time(); OUT = []
def say(*m):
    s = " ".join(str(x) for x in m); print(s, flush=True); OUT.append(s)
L = list(range(1, 7)); GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str)
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation"]], ignore_index=True)
raw["sample_id"] = raw.sample_id.astype(str); raw = raw.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index(); raw["experiment_id"] = raw.experiment_id.astype(str)
y = fr.label.astype(int).values; ex = raw.experiment_id.values; t = raw.relative_time_s.values.astype(float); N = len(y)
tr = raw.tr_method.values.astype(int); bc = raw.bat_class.values.astype(int); soc = raw.soc_status.values.astype(int)
groups = pd.Series(np.arange(N)).groupby(ex).indices; groups = {e: ix[np.argsort(t[ix], kind="stable")] for e, ix in groups.items()}
ue = np.array(sorted(groups))
def slope30(x, tt):
    j = np.searchsorted(tt, tt - 30.0, side="left"); return (x - x[j]) / np.maximum(tt - tt[j], 5.0)
cmax = np.maximum.accumulate; cmin = np.minimum.accumulate
FE = {}   # 피처 풀 (행 순서는 fr 과 동일, 실험별 인과 계산)
def put(name, fn):
    out = np.zeros(N)
    for e, ix in groups.items(): out[ix] = fn(raw.iloc[ix], t[ix])
    FE[name] = out
put("thc_atmp_cm", lambda d, tt: cmax((d.thermal_t - d.atmp_t).values)); put("th_cm", lambda d, tt: cmax(d.thermal_t.values))
put("t1rise_cm", lambda d, tt: cmax((d.meas_t1 - d.meas_t1.iloc[0]).values)); put("t2t1_cmin", lambda d, tt: cmin((d.meas_t2 - d.meas_t1).values))
put("th_sl30_cm", lambda d, tt: cmax(slope30(d.thermal_t.values, tt))); put("t1_sl30_cm", lambda d, tt: cmax(slope30(d.meas_t1.values, tt)))
put("prise_cm", lambda d, tt: cmax((d.meas_p - d.meas_p.iloc[0]).values)); put("t1_cm", lambda d, tt: cmax(d.meas_t1.values)); put("t2_cm", lambda d, tt: cmax(d.meas_t2.values))
put("th_t1_cmin", lambda d, tt: cmin((d.thermal_t - d.meas_t1).values)); put("vr_cmin", lambda d, tt: cmin((d.meas_v / max(d.meas_v.iloc[0], 1e-6)).values))
put("t2rise_cm", lambda d, tt: cmax((d.meas_t2 - d.meas_t2.iloc[0]).values)); put("th_rise_cm", lambda d, tt: cmax((d.thermal_t - d.thermal_t.iloc[0]).values))
POOL = list(FE); XP = np.column_stack([FE[k] for k in POOL])
rng = np.random.default_rng(0)
def fold_assign(seed):
    r = np.random.default_rng(seed); p = r.permutation(len(ue)); m = {e: i % 5 for i, e in zip(range(len(ue)), ue[p])}; return np.array([m[e] for e in ex])
def boot_ci(d, B=2000):
    r = np.random.default_rng(1); m = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(B)]); return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
def exp_acc(pred, yy, ee):
    return pd.Series((pred == yy).astype(float)).groupby(ee).mean()
lgb_p = np.load(ROOT / "outputs/preds" / f"{a.oof}.npy"); lgb_lab = F.postprocess(fr, lgb_p, 0.7) if lgb_p.ndim > 1 else lgb_p
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=300, C=1.0))
def run_exp1():
    say("\n## 실험 1: L2 vs L3 저용량 모델 (L2/L3 프레임만, 방식별), 피처 풀 %d개" % len(POOL))
    FIX = [POOL.index(k) for k in ("thc_atmp_cm", "t1rise_cm", "t2t1_cmin")]
    for m, nm in ((2, "과충전"), (1, "가열")):
        sel = np.where((tr == m) & np.isin(y, [2, 3]))[0]; yb = (y[sel] == 3).astype(int); ee = ex[sel]; nexp = len(np.unique(ee))
        accs = {k: [] for k in "abcd"}; freq = np.zeros(len(POOL)); nf = 0; scores = {k: [] for k in "abcd"}
        for seed in (0, 1, 2):
            fo = fold_assign(seed)[sel]; pa = np.zeros(len(sel), int); pb = pa.copy(); pc = pa.copy()
            for k in range(5):
                tri, tei = np.where(fo != k)[0], np.where(fo == k)[0]
                if len(np.unique(yb[tri])) < 2: continue
                # (a) 중첩 선택
                inner = fold_assign(100 + seed * 7 + k)[sel][tri] % 3; chosen = []; best = 0.0
                for _ in range(3):
                    cand = None
                    for j in range(len(POOL)):
                        if j in chosen: continue
                        cols = chosen + [j]; sc = []
                        for q in range(3):
                            a_, b_ = tri[inner != q], tri[inner == q]
                            if len(np.unique(yb[a_])) < 2 or len(b_) == 0: continue
                            mdl = lr().fit(XP[sel][a_][:, cols], yb[a_]); sc.append(f1_score(yb[b_], mdl.predict(XP[sel][b_][:, cols]), zero_division=0))
                        s = np.mean(sc) if sc else 0
                        if cand is None or s > cand[1]: cand = (j, s)
                    if cand is None or cand[1] - best < 0.002: break
                    chosen.append(cand[0]); best = cand[1]
                if not chosen: chosen = [FIX[0]]
                for j in chosen: freq[j] += 1
                nf += 1
                pa[tei] = lr().fit(XP[sel][tri][:, chosen], yb[tri]).predict(XP[sel][tei][:, chosen])
                pb[tei] = lr().fit(XP[sel][tri][:, FIX], yb[tri]).predict(XP[sel][tei][:, FIX])
                x = XP[sel][:, POOL.index("thc_atmp_cm")]; ths = np.unique(np.quantile(x[tri], np.linspace(0.02, 0.98, 60)))
                f1s = [f1_score(yb[tri], (x[tri] >= th).astype(int), zero_division=0) for th in ths]; th = ths[int(np.argmax(f1s))]; pc[tei] = (x[tei] >= th).astype(int)
            pd_ = (lgb_lab[sel] >= 3).astype(int)
            for k, p in zip("abcd", (pa, pb, pc, pd_)):
                accs[k].append(exp_acc(p, yb, ee)); scores[k].append((f1_score(yb, p, zero_division=0), balanced_accuracy_score(yb, p)))
        say(f"\n[{nm}] 실험 {nexp}개, L2/L3 프레임 {len(sel)}")
        for k, lab_ in zip("abcd", ("(a) 중첩 선택 로지스틱", "(b) 고정 3개 로지스틱", "(c) 단일 임계 thermal-atmp", "(d) 센서 LightGBM OOF")):
            s = np.mean(scores[k], 0); say(f"  {lab_:26s} L3 F1 {s[0]:.3f} | 균형정확도 {s[1]:.3f}")
        dacc = pd.concat(accs["d"], axis=1).mean(1)
        for k, lab_ in (("a", "(a)"), ("b", "(b)"), ("c", "(c)")):
            d = (pd.concat(accs[k], axis=1).mean(1) - dacc).values; mu, lo, hi = boot_ci(d)
            say(f"  {lab_}-(d) 실험별 정확도 차 {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] | 개선 {(d > 0.005).sum()} 악화 {(d < -0.005).sum()} / {len(d)}개 실험")
        say("  중첩 선택 피처 빈도: " + ", ".join(f"{POOL[j]} {freq[j] / max(nf, 1):.2f}" for j in np.argsort(-freq)[:6]))
def run_exp3():
    say("\n## 실험 3: SOC 100% 실험의 진행 속도 (분석용, 실험 전체 통계)")
    rows = []
    for e, ix in groups.items():
        tt = t[ix]; t1 = raw.meas_t1.values[ix].astype(float); yy = y[ix]; cm = np.maximum.accumulate(t1)
        if not (cm >= 45).any(): continue
        j = int(np.argmax(cm >= 45)); t45 = tt[j]; r = {}
        for w in (60, 120):
            k = np.searchsorted(tt, t45 + w, side="left"); k = min(k, len(tt) - 1); r[w] = (t1[k] - t1[j]) / max(tt[k] - t45, 5.0)
        l2 = tt[yy >= 2][0] if (yy >= 2).any() else np.nan; l3 = tt[yy >= 3][0] if (yy >= 3).any() else np.nan
        rows.append(dict(e=e, soc=int(soc[ix[0]]), tr=int(tr[ix[0]]), bc=int(bc[ix[0]]), t45=t45, r60=r[60], r120=r[120], l2dur=l3 - l2, err=1 - (lgb_lab[ix] == yy).mean()))
    D = pd.DataFrame(rows).dropna(); D["ll2"] = np.log(D.l2dur.clip(lower=5))
    say(f"실험 {len(D)}개 | 오류율 SOC1(50%) {D[D.soc == 1].err.mean():.3f} vs SOC2(100%) {D[D.soc == 2].err.mean():.3f} | L2 지속 중앙 SOC1 {D[D.soc == 1].l2dur.median():.0f}s SOC2 {D[D.soc == 2].l2dur.median():.0f}s")
    say("형태x방식별 SOC 오류율(50%/100%): " + ", ".join(f"bc{b}tr{m}: {g[g.soc == 1].err.mean():.3f}/{g[g.soc == 2].err.mean():.3f}" for (b, m), g in D.groupby(["bc", "tr"])))
    def coef(d, cols):
        X = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols]); b = np.linalg.lstsq(X, d.err.values, rcond=None)[0]; return b[1]
    D["soc2"] = (D.soc == 2).astype(float); D["tr2"] = (D.tr == 2).astype(float)
    for nm, cols in (("SOC + 방식", ["soc2", "tr2"]), ("SOC + log(L2 지속) + 방식", ["soc2", "ll2", "tr2"])):
        r = np.random.default_rng(2); bs = [coef(D.iloc[r.integers(0, len(D), len(D))], cols) for _ in range(2000)]
        say(f"  오류율 ~ {nm}: SOC 계수 {coef(D, cols):+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    for c in ("r60", "r120"):
        rho, p = spearmanr(D[c], D.l2dur); say(f"  Spearman({c}, L2 지속시간) {rho:+.2f} (p={p:.3g})  | SOC별 {c} 중앙 {D[D.soc == 1][c].median():.3f}/{D[D.soc == 2][c].median():.3f} ℃/s")
def run_exp4():
    say("\n## 실험 4: 단계 시작 시각 오차 (예측 시작 - 정답 시작, 초)")
    rows = []
    for e, ix in groups.items():
        tt = t[ix]
        for k in range(2, 7):
            ta = tt[y[ix] >= k][0] if (y[ix] >= k).any() else np.nan; tp = tt[lgb_lab[ix] >= k][0] if (lgb_lab[ix] >= k).any() else np.nan
            if not np.isnan(ta): rows.append(dict(e=e, bc=int(bc[ix[0]]), tr=int(tr[ix[0]]), k=k, err=tp - ta if not np.isnan(tp) else np.nan))
    D = pd.DataFrame(rows); BN, TN = {1: "원통", 2: "파우치", 3: "각형"}, {1: "가열", 2: "과충전"}
    say(f"예측이 한 번도 도달하지 못한 (실험,경계) {int(D.err.isna().sum())}개 / {len(D)}개 (경계 {D[D.err.isna()].k.value_counts().sort_index().to_dict()})")
    for (b, m), g in D.groupby(["bc", "tr"]):
        say(f"[{BN[b]}·{TN[m]}] " + " | ".join(f"L{k}: 절대중앙 {g[g.k == k].err.abs().median():.0f}s ±10s {np.mean(g[g.k == k].err.abs() <= 10):.2f} ±30s {np.mean(g[g.k == k].err.abs() <= 30):.2f}" for k in range(2, 7) if g[g.k == k].err.notna().any()))
    sc = D[D.k >= 3].groupby("e").err.apply(lambda s: s.abs().mean()).dropna(); mu, lo, hi = boot_ci(sc.values)
    say(f"제안 지표: 경계 L3~L6 평균 절대 시작시각 오차(실험별 평균의 평균) {mu:.1f}s [{lo:.1f}, {hi:.1f}] (프레임 F1 과 달리 파우치 L4/L5 처럼 짧은 단계도 시간 오차로 비교 가능)")
def run_exp2():
    say("\n## 실험 2: 가스 상대화 변형 (로지스틱, 6단계 OOF Macro-F1 후처리)")
    L1 = y == 1
    for g in ("meas_n2o", "meas_no", "meas_no2", "meas_co2"):
        v = np.log1p(raw[g].values.astype(float)); vr = v.copy()
        for e, ix in groups.items(): vr[ix] = v[ix] - v[ix[0]]
        eta = lambda z: pd.Series(z[L1]).groupby(ex[L1]).transform("mean").var() / max(pd.Series(z[L1]).var(), 1e-12)
        say(f"  eta^2(L1 프레임 실험정체성 비율) {g}: 절대 {eta(v):.2f} -> 첫 관측 대비 {eta(vr):.2f}")
    G = np.log1p(raw[GAS].values.astype(float)); gs = np.log1p(raw[GAS].sum(axis=1).values.astype(float))
    def rel(z):
        o = z.copy()
        for e, ix in groups.items(): o[ix] = z[ix] - z[ix[0]]
        return o
    def cmx(z):
        o = z.copy()
        for e, ix in groups.items(): o[ix] = np.maximum.accumulate(z[ix])
        return o
    base = np.column_stack([FE["vr_cmin"], raw.meas_t1, raw.meas_t2, raw.thermal_t, np.log1p(raw.meas_p), np.minimum(raw.atmp_t, 60), FE["t1_cm"], FE["t2_cm"], FE["th_cm"], FE["t1rise_cm"], FE["th_rise_cm"], FE["t1_sl30_cm"], FE["th_sl30_cm"],
                            (tr == 2), (bc == 1), (bc == 2), (bc == 3)]).astype(float)
    Gr = np.column_stack([rel(G[:, i]) for i in range(10)])
    V = {"V0 가스 없음": base, "V1 가스합 절대": np.column_stack([base, gs]), "V2 가스10종 절대": np.column_stack([base, G]), "V3 가스합 첫관측대비": np.column_stack([base, rel(gs)]),
         "V4 가스10종 첫관측대비": np.column_stack([base, Gr]), "V5 가스합 상대 누적최대": np.column_stack([base, cmx(rel(gs))])}
    sub = np.arange(N) % 2 == 0; res = {}; accs = {}
    for nm, X in V.items():
        f = []; ac = []
        for seed in (0, 1):
            fo = fold_assign(seed); P = np.zeros((N, 6))
            for k in range(5):
                tri = np.where((fo != k) & sub)[0]; tei = np.where(fo == k)[0]
                m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=200, C=0.5)).fit(X[tri], y[tri]); P[np.ix_(tei, m.classes_ - 1)] = m.predict_proba(X[tei])
            pl = F.postprocess(fr, P, 0.7); f.append(f1_score(y, pl, labels=L, average="macro", zero_division=0)); ac.append(exp_acc(pl, y, ex))
        res[nm] = np.mean(f); accs[nm] = pd.concat(ac, axis=1).mean(1)
        say(f"  {nm:22s} F1 {res[nm]:.4f}" + ("" if nm.startswith("V0") else (lambda d: f" | V0 대비 {d.mean():+.4f} [{boot_ci(d.values)[1]:+.4f}, {boot_ci(d.values)[2]:+.4f}] (실험별 정확도)")(accs[nm] - accs["V0 가스 없음"])))
for k in a.only.split(","):
    {"1": run_exp1, "3": run_exp3, "4": run_exp4, "2": run_exp2}[k.strip()](); say(f"  (경과 {time.time() - T0:.0f}s)")
open(ROOT / "outputs" / "DA_REPORT.md", "w").write("# DA(CPU) 분석 결과\n\n```\n" + "\n".join(OUT) + "\n```\n")
