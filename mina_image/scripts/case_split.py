"""실험별 오류 사례 분류(최종 LightGBM OOF). 실험마다 L2<->L3 오류 행 수, L3 시작 시각 오차(예측 첫 L3 - 실제 L3), 유형(늦게/일찍/정확), 실험 속성(조건, 가열속도, 레짐, 길이)을 계산하고
어떤 속성이 실패(오류율)와 연관되는지 본다. 분석용이라 실험 전체 통계(속성)를 쓴다(모델 입력 아님)."""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import DATA, ROOT, table
sys.path.insert(0, str(ROOT / "ts_model")); import features as F
tb = table(); lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", usecols=["sample_id", "experiment_id", "relative_time_s", "meas_t1", "meas_v", "meas_p", "thermal_t", "atmp_t", "bat_class", "bat_cc", "soc_status", "tr_method"]) for s in ["train", "validation"]])
raw["sample_id"] = raw.sample_id.astype(str); raw = raw.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index(); raw["experiment_id"] = raw.experiment_id.astype(str)
p = F.postprocess(fr, np.load(ROOT / "outputs/preds/final_lgb_oof.npy"), 0.7)
rows = []
for e, ix in pd.Series(np.arange(len(y))).groupby(raw.experiment_id.values).indices.items():
    ix = ix[np.argsort(raw.relative_time_s.values[ix], kind="stable")]; t = raw.relative_time_s.values[ix].astype(float); yy = y[ix]; pp = p[ix]; t1 = raw.meas_t1.values[ix].astype(float)
    e23 = int((((yy == 2) & (pp == 3)) | ((yy == 3) & (pp == 2))).sum()); tot = int((yy != pp).sum())
    ons = t[yy >= 3].min() if (yy >= 3).any() else np.nan; pon = t[pp >= 3].min() if (pp >= 3).any() else np.nan
    cm = np.maximum.accumulate(t1); t45 = t[np.argmax(cm >= 45)] if (cm >= 45).any() else np.nan
    l2on = t[yy >= 2].min() if (yy >= 2).any() else np.nan
    # L3 구간 내부에서 모델이 L2 로 되돌아간 프레임(late 후 flicker 아님: 후처리 이후 예측 감소 수)
    l3 = yy == 3; reverts = int(((pp[l3][1:] < pp[l3][:-1])).sum()) if l3.sum() > 1 else 0
    rows.append(dict(exp=e[:8], cls=int(raw.bat_class.values[ix][0]), tr=int(raw.tr_method.values[ix][0]), cc=float(raw.bat_cc.values[ix][0]), soc=int(raw.soc_status.values[ix][0]), n=len(ix), err=tot, err23=e23, err_rate=tot / len(ix),
                     onset_err=pon - ons, t45=t45, heat=(45 - t1[0]) / max(t45, 5), l2dur=ons - l2on, t1_at_l3=t1[yy >= 3][0] if (yy >= 3).any() else np.nan, jump=int(np.any(np.diff(yy) > 1)), reverts=reverts))
d = pd.DataFrame(rows); d["유형"] = np.where(d.onset_err.abs() <= 30, "정확(±30s)", np.where(d.onset_err > 0, "늦게(L3 늦음)", "일찍(L3 이름)"))
d["regime_B"] = (d.t1_at_l3 >= 65).astype(int)
for tr_, nm in [(1, "가열"), (2, "과충전")]:
    g = d[d.tr == tr_]
    print(f"\n===== {nm}: 실험 {len(g)}개 | L2<->L3 오류 {int(g.err23.sum())}행 (전체 오류의 {g.err23.sum() / g.err.sum():.0%}) =====")
    print("유형별 실험 수 / 그 실험들의 L2<->L3 오류 합:"); print(g.groupby("유형").agg(실험=("exp", "size"), 오류합=("err23", "sum"), 평균오차초=("onset_err", "mean")).round(1).to_string())
    print(f"오류 상위 5개 실험이 L2<->L3 오류의 {g.err23.nlargest(5).sum() / g.err23.sum():.0%}")
    print("상위 8개 실험:"); print(g.sort_values("err23", ascending=False).head(8)[["exp", "cls", "cc", "soc", "n", "err23", "onset_err", "t45", "heat", "l2dur", "t1_at_l3", "regime_B", "jump"]].round(2).to_string(index=False))
    rr = []
    for a in ["cls", "cc", "soc", "n", "t45", "heat", "l2dur", "t1_at_l3", "regime_B", "jump"]:
        ok = g[[a, "err_rate"]].dropna()
        if ok[a].nunique() > 1: s = spearmanr(ok[a], ok["err_rate"]); rr.append((a, s[0], s[1]))
    print("속성과 오류율의 Spearman 상관:", {a: f"{r:+.2f}(p={p_:.2f})" for a, r, p_ in sorted(rr, key=lambda x: -abs(x[1]))})
