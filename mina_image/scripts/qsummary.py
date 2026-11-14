"""신경망 실험 결과 요약: outputs/preds/ft_mm_<run>_oof_gated.npy 전부를 모아 합산 OOF / 500행 상한 가중 F1 / LightGBM 블렌드 / 기준 대비 실험 단위 부트스트랩 CI 를 표로 만들고
outputs/REPORT.md 에 쓴다. 비 LightGBM 최고 모델의 Test CSV(단독 gated)를 자동 생성한다(final_lgb 와 별개, 이미 있으면 건너뜀). 사용: python scripts/qsummary.py [--base all_hdtr]"""
import argparse, subprocess, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
ap = argparse.ArgumentParser(); ap.add_argument("--base", default="all_hdtr"); a = ap.parse_args()
L = list(range(1, 7)); tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str)
y = fr.label.astype(int).values; ex = fr.experiment_id.values; ue = np.unique(ex); eidx = {e: np.where(ex == e)[0] for e in ue}
cnt = pd.Series(ex).map(pd.Series(ex).value_counts()).values.astype(float); wcap = np.minimum(1.0, 500.0 / cnt)
P = ROOT / "outputs/preds"; f1 = lambda p, w=None: f1_score(y, p, labels=L, average="macro", zero_division=0, sample_weight=w)
lg = np.load(P / "final_lgb_oof.npy"); lg_lab = F.postprocess(fr, lg, 0.7)
from common import DATA
_r = pd.concat([pd.read_csv(DATA / sp / f"{sp}.csv", usecols=["sample_id", "experiment_id", "relative_time_s", "thermal_t", "atmp_t", "tr_method"], dtype={"sample_id": "string", "experiment_id": "string"}) for sp in ["train", "validation"]])
_r["sample_id"] = _r.sample_id.astype(str); _r = _r.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index(); _t = _r.relative_time_s.values.astype(float)
tha = (_r.thermal_t - _r.atmp_t).values.astype(float); tha_cm = np.zeros(len(y)); oc = _r.tr_method.values == 2; fold = fr.fold.values
for _e, _ix in pd.Series(np.arange(len(y))).groupby(_r.experiment_id.astype(str).values).indices.items():
    _ix = _ix[np.argsort(_t[_ix], kind="stable")]; tha_cm[_ix] = np.maximum.accumulate(tha[_ix])
def pick_theta(m):
    v = tha_cm[m]; grid = np.unique(np.nanpercentile(v, np.linspace(2, 98, 33))); yy = (y[m] == 3).astype(int); best = (-1, grid[0])
    for g in grid:
        f = f1_score(yy, (v >= g).astype(int), zero_division=0); best = max(best, (f, g), key=lambda z: z[0])
    return best[1]
def apply_override(G):          # 과충전 L3 규칙 덮어쓰기(final_lgb 와 동일, 임계는 학습 fold 의 과충전 L2/L3 프레임에서 선택)
    O = G.copy()
    for k in range(5):
        th = pick_theta((fold != k) & oc & np.isin(y, [2, 3])); rr = np.where((fold == k) & oc)[0]; Z = O[rr]; l3 = tha_cm[rr] >= th; p23 = Z[:, 1] + Z[:, 2]; sel = (Z.argmax(1) >= 1) & (Z.argmax(1) <= 2)
        Z[sel, 1] = np.where(l3[sel], 0.0, p23[sel]); Z[sel, 2] = np.where(l3[sel], p23[sel], 0.0); O[rr] = Z
    return O
def start_err(pl):              # 경계 L3~L6 평균 절대 시작시각 오차(초), 실험별 평균의 평균
    v = []
    for e, ix in eidx.items():
        ix = ix[np.argsort(_t[ix], kind="stable")]; tt = _t[ix]; er = []
        for k in range(3, 7):
            if (y[ix] >= k).any() and (pl[ix] >= k).any(): er.append(abs(tt[pl[ix] >= k][0] - tt[y[ix] >= k][0]))
        if er: v.append(np.mean(er))
    return float(np.mean(v))
rows, labs = [], {}
for f in sorted(P.glob("ft_mm_*_oof_gated.npy")):
    run = f.name[len("ft_mm_"):-len("_oof_gated.npy")]; G = np.load(f)
    if G.shape != lg.shape or not (P / f"ft_mm_{run}_test.npy").exists(): continue
    pg = F.postprocess(fr, G, 0.7); pb = F.postprocess(fr, 0.7 * lg + 0.3 * G, 0.7); labs[run] = pg
    po = F.postprocess(fr, apply_override(G), 0.7); labs[run + "+ovr"] = po
    rows.append(dict(run=run, gated=f1(pg), cap500=f1(pg, wcap), blend30=f1(pb), ovr=f1(po), serr=start_err(pg), mtime=f.stat().st_mtime))
D = pd.DataFrame(rows).sort_values("gated", ascending=False).reset_index(drop=True)
rng = np.random.default_rng(0); B = 300; samp = [np.concatenate([eidx[e] for e in rng.choice(ue, len(ue), replace=True)]) for _ in range(B)]
def boot(p, q): d = np.array([f1_score(y[i], p[i], labels=L, average="macro", zero_division=0, sample_weight=wcap[i]) - f1_score(y[i], q[i], labels=L, average="macro", zero_division=0, sample_weight=wcap[i]) for i in samp]); return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)
base = a.base if a.base in labs else D.run[0]
D["vs_base"] = [("-" if r == base else "%+.4f [%+.4f, %+.4f]" % boot(labs[r], labs[base])) for r in D.run]
best = D.run[0]; vl = boot(labs[best], lg_lab)
csv_ok = ROOT / "outputs/submissions" / f"mm_{best}_gated.csv"
if not csv_ok.exists(): subprocess.run([sys.executable, "scripts/make_nn_csv.py", best], check=False, capture_output=True)
lines = [f"# 자동 보고서 ({time.strftime('%Y-%m-%d %H:%M')} 서버시각, 사용자 시각 = -3h)", "", f"기준 모델 `{base}`, 비교는 500행 상한 가중 F1의 실험 단위 부트스트랩({B}회) 차이 [95% CI]. LightGBM `final_lgb`: OOF {f1(lg_lab):.4f} / 500행 가중 {f1(lg_lab, wcap):.4f}, Test 0.8680.", "",
         "| run | gated OOF | 500행 가중 | LightGBM 블렌드 w=0.3 | 기준 대비 | +과충전 규칙 | 시작시각 오차(s) |", "|---|---|---|---|---|---|---|"]
lines += [f"| {r.run} | {r.gated:.4f} | {r.cap500:.4f} | {r.blend30:.4f} | {r.vs_base} | {r.ovr:.4f} | {r.serr:.1f} |" for r in D.itertuples()]
lines += ["", f"**비 LightGBM 최고: `{best}`** (gated OOF {D.gated[0]:.4f}). `final_lgb` 대비 {vl[0]:+.4f} [{vl[1]:+.4f}, {vl[2]:+.4f}] (CI 가 0을 포함하면 구분 불가).",
          f"올릴 CSV(NN 단독, final_lgb 와 별개): `outputs/submissions/mm_{best}_gated.csv`" + ("" if csv_ok.exists() else " (생성됨)"),
          "", "참고: 블렌드(w=0.3) CSV 는 final_lgb 와 99% 이상 같아 Test 차이가 거의 없음. 다음 파일도 확인: `outputs/DA_REPORT.md`(CPU 분석), `outputs/queue/done.txt`(실행 이력)."]
ts = subprocess.run([sys.executable, "scripts/test_sim.py", "final_lgb_oof"] + [f"ft_mm_{r}_oof_gated" for r in D.run[:3]], capture_output=True, text=True, env=dict(__import__("os").environ, PYTHONPATH="scripts")).stdout.strip().splitlines()
lines += ["", "가상 Test(Test 그룹 구성 재현, 상위 3개 + final_lgb):", "```"] + [l for l in ts if "풀링" in l] + ["```"]
open(ROOT / "outputs/REPORT.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
