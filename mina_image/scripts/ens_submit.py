"""센서(시계열) LightGBM+XGBoost 앙상블 + 이미지 확률(선정 백본) 융합 -> Validation 점수 보고 -> 제출 CSV.

한 번 실행으로 끝까지:
 1) 공식 프로토콜(Train 학습 -> Validation 평가)로 각 설정의 Macro F1 을 보고
      설정 = {ts(센서만), light(+img_exp,img_p456)} x {lgb, xgb, lgb+xgb} x blend_w
      평가셋 = cap500(Test 와 같은 500행 상한 다운샘플) 과 full(5초 그대로)
 2) cap500 앙상블 F1 이 가장 높은 설정을 고른다(Validation 으로 설정을 고르므로 보고 점수는 약간 낙관적, 로그에 명시)
 3) 그 설정으로 Train+Validation 전체를 다시 학습해 Test 예측 -> 후처리(확신 cummax) -> 제출 CSV 검증·저장
이미지 확률: 학습 행=5-fold OOF, Test 행=fold 모델 평균(image_probs.parquet). 학습 행의 OOF 는 Validation 실험이 섞인 fold 모델에서 나왔으므로
Train->Val 점수에 아주 약한 누수가 있을 수 있다(fusion_cv --protocol trainval 과 같은 조건).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from common import DATA, ROOT
from fusion_lib import IMG_VARIANTS, add_image_cols

ap = argparse.ArgumentParser()
ap.add_argument("--ts_dir", type=Path, default=ROOT / "ts_model")
ap.add_argument("--data", type=Path, default=DATA)
ap.add_argument("--models", nargs="+", default=["ft_dinov2_b"], help="image_probs.parquet 의 백본 태그(여러 개면 확률 평균)")
ap.add_argument("--probs", default="image_probs")
ap.add_argument("--variants", default="ts,light")
ap.add_argument("--blend_ws", default="0,0.1,0.2,0.3")
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--xgb_device", default="auto", choices=["auto", "cuda", "cpu"])
ap.add_argument("--n_jobs", type=int, default=-1, help="CPU 스레드(다른 작업과 CPU 를 나눌 땐 8 등으로 제한)")
ap.add_argument("--gas", default="all", choices=["all", "sum"], help="sum=개별 가스 10종을 0 으로 만들고 가스합만 meas_co2 칸에 유지(가스 정리). OOF 에서 +0.0087(4/5 fold)")
ap.add_argument("--name", default="ens_lgb_xgb")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fast", action="store_true", help="트리 수 절반(스크리닝)")
ap.add_argument("--smoke", action="store_true", help="실험 일부·트리 30개로 배선만 확인(제출 파일은 _smoke 이름)")
ap.add_argument("--skip_final", action="store_true", help="Validation 보고까지만")
a = ap.parse_args()
sys.path.insert(0, str(a.ts_dir))
import features as F  # 시계열 팀 코드

T0 = time.time()


def log(*m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *m, flush=True)


L = list(range(1, 7))
BASE0 = list(F.BASE_SIGNALS)
variants = [v for v in a.variants.split(",") if v]
use_img = bool(a.models)
blend_ws = [float(x) for x in a.blend_ws.split(",")] if use_img else [0.0]
assert all(v == "ts" or v in IMG_VARIANTS for v in variants), f"변형은 ts/light/probs6 만 지원: {variants}"
if any(v != "ts" for v in variants) or any(w > 0 for w in blend_ws):
    assert use_img, "이미지 변형/blend 에는 --models 가 필요"


# ---------------------------------------------------------------- 모델
def make_lgb():
    m = F.build_model()  # 팀 val0.9195 모델과 동일 하이퍼
    m.set_params(n_jobs=a.n_jobs)
    n = 30 if a.smoke else (250 if a.fast else None)
    if n:
        m.set_params(n_estimators=n, learning_rate=0.06 if a.fast and not a.smoke else 0.1)
    return m


_xgb_dev = {"v": None}


def make_xgb(device):
    import xgboost as xgb
    n = 30 if a.smoke else (250 if a.fast else 500)
    return xgb.XGBClassifier(
        n_estimators=n, learning_rate=0.06 if a.fast else 0.03, max_depth=7, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.3, reg_lambda=1.0, tree_method="hist", device=device,
        objective="multi:softprob", num_class=6, random_state=a.seed, n_jobs=a.n_jobs, verbosity=0)


def fit_xgb(X, y):
    sw = compute_sample_weight("balanced", y)
    devs = [a.xgb_device] if a.xgb_device != "auto" else ([_xgb_dev["v"]] if _xgb_dev["v"] else ["cuda", "cpu"])
    for dv in devs:
        try:
            m = make_xgb(dv).fit(X, y - 1, sample_weight=sw)
            _xgb_dev["v"] = dv
            return m
        except Exception as e:  # GPU 미지원 등
            log(f"xgboost device={dv} 실패({type(e).__name__}: {str(e)[:80]}) -> 다음 후보")
    raise RuntimeError("xgboost 학습 실패")


def fit_all(X, y):
    t = time.time()
    lg = make_lgb().fit(X, y)
    log(f"   lgb 학습 {time.time() - t:.0f}s")
    t = time.time()
    xg = fit_xgb(X, y)
    log(f"   xgb 학습 {time.time() - t:.0f}s (device={_xgb_dev['v']})")
    return lg, xg


def probas(models, X):
    lg, xg = models
    P = {"lgb": lg.predict_proba(X), "xgb": xg.predict_proba(X)}
    P["ens"] = (P["lgb"] + P["xgb"]) / 2
    return P


# ---------------------------------------------------------------- 데이터
def read(split):
    d = pd.read_csv(a.data / split / f"{split}.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    d["sample_id"], d["experiment_id"] = d.sample_id.astype(str), d.experiment_id.astype(str)
    d["split"] = split
    return d


lab = pd.concat([read("train"), read("validation")], ignore_index=True)
tst = read("test")
if a.gas == "sum":
    _G = list(F.GAS)
    for _d in (lab, tst):
        _s = _d[_G].sum(axis=1)
        for _g in _G: _d[_g] = 0.0
        _d["meas_co2"] = _s
if a.smoke:
    keep = pd.concat([lab[lab.split == s].experiment_id.drop_duplicates().head(n) for s, n in (("train", 8), ("validation", 3))])
    lab = lab[lab.experiment_id.isin(keep)].reset_index(drop=True)
    tst = tst[tst.experiment_id.isin(tst.experiment_id.drop_duplicates().head(3))].reset_index(drop=True)
if use_img:
    lab, tst = add_image_cols(lab, a.probs, a.models), add_image_cols(tst, a.probs, a.models)
    assert lab.img_exp.notna().all(), "라벨 행에 이미지 OOF 확률이 없음 (build_image_features.py 먼저)"
    assert tst.img_exp.notna().all(), f"Test 행 이미지 확률 없음: {a.models} 의 *_test.npy 필요"
lab = lab.reset_index(drop=True)
tr_mask, va_mask = (lab.split == "train").values, (lab.split == "validation").values
log(f"학습(Train) {tr_mask.sum()}행/{lab[tr_mask].experiment_id.nunique()}실험 | Validation {va_mask.sum()}행/{lab[va_mask].experiment_id.nunique()}실험 | "
    f"Test {len(tst)}행/{tst.experiment_id.nunique()}실험 | 이미지 백본 {a.models if use_img else '없음'}")

vfull = lab[va_mask].reset_index(drop=True)
vcap = F.cap_rows(lab[va_mask]).reset_index(drop=True)
evals = {"cap500": vcap, "full": vfull}


def feats(frame, extra):
    F.BASE_SIGNALS = BASE0 + extra
    F.SIGNALS = F.BASE_SIGNALS + F.DERIVED
    return F.build_features(frame.reset_index(drop=True))


def f1(y, p):
    return f1_score(y, p, labels=L, average="macro", zero_division=0)


# ---------------------------------------------------------------- 1) Train -> Validation 보고
log("=" * 70, "\n1) 공식 프로토콜: Train 으로 학습 -> Validation 평가")
rows, store = [], {}
for v in variants:
    extra = IMG_VARIANTS.get(v, [])
    log(f"[{v}] feature 계산")
    Xall = feats(lab, extra)
    cols = list(Xall.columns)
    log(f"[{v}] feature {len(cols)}개, 학습 {tr_mask.sum()}행 -> 학습")
    models = fit_all(Xall[tr_mask], lab.label.values[tr_mask].astype(int))
    for en, fr in evals.items():
        Xv = Xall[va_mask].reset_index(drop=True) if en == "full" else feats(fr, extra)[cols]
        y = fr.label.astype(int).values
        P = probas(models, Xv)
        store[(v, en)] = (P, fr)
        for mn, Pm in P.items():
            ws = blend_ws if (use_img and mn == "ens" and v == "ts") else [0.0]
            for w in ws:
                Pb = (1 - w) * Pm + w * fr[[f"img_p{k}" for k in L]].values if w > 0 else Pm
                pred = F.postprocess(fr, Pb, a.tau)
                rows.append(dict(variant=v, model=mn, blend_w=w, eval=en, f1_post=f1(y, pred), f1_raw=f1(y, F.LABELS[Pb.argmax(1)]),
                                 adj=float((np.abs(pred - y) <= 1).mean()), per_class=[round(float(x), 3) for x in f1_score(y, pred, labels=L, average=None, zero_division=0)]))
    r = pd.DataFrame(rows)
    log(f"[{v}] 완료. cap500: " + ", ".join(f"{m}={r[(r.variant == v) & (r.model == m) & (r.blend_w == 0) & (r['eval'] == 'cap500')].f1_post.iloc[0]:.4f}" for m in ("lgb", "xgb", "ens")))

R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n===== Validation Macro F1 (후처리 = 확신 cummax, tau=%.2f) =====" % a.tau)
for en in evals:
    print(f"\n--- 평가셋 {en} ---")
    print(R[R['eval'] == en].drop(columns=["eval"]).sort_values("f1_post", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
out_dir = ROOT / "outputs" / "logs"
out_dir.mkdir(parents=True, exist_ok=True)
R.to_csv(out_dir / f"{a.name}{'_smoke' if a.smoke else ''}_val.csv", index=False)

cand = R[(R['eval'] == "cap500") & (R.model == "ens")].sort_values("f1_post", ascending=False)
best = cand.iloc[0]
BV, BW = best.variant, float(best.blend_w)
print(f"\n>>> 선택 설정(cap500 앙상블 최고): variant={BV} blend_w={BW}  VALIDATION Macro F1 = {best.f1_post:.4f} "
      f"(full {R[(R['eval'] == 'full') & (R.model == 'ens') & (R.variant == BV) & (R.blend_w == BW)].f1_post.iloc[0]:.4f})")
print("    ※ 설정을 Validation 으로 골랐으므로 이 점수는 약간 낙관적. 센서 단독 ens 기준선 = "
      f"{R[(R['eval'] == 'cap500') & (R.model == 'ens') & (R.variant == 'ts') & (R.blend_w == 0)].f1_post.iloc[0]:.4f}")
if a.skip_final:
    sys.exit(0)

# ---------------------------------------------------------------- 2) Train+Validation 재학습 -> Test -> CSV
log("=" * 70, "\n2) 최종: Train+Validation 으로 재학습 -> Test 예측")
extra = IMG_VARIANTS.get(BV, [])
Xall = feats(lab, extra)
cols = list(Xall.columns)
models = fit_all(Xall, lab.label.values.astype(int))
tst = tst.reset_index(drop=True)
Pte = probas(models, feats(tst, extra)[cols])["ens"]
if BW > 0:
    Pte = (1 - BW) * Pte + BW * tst[[f"img_p{k}" for k in L]].values
pred = F.postprocess(tst, Pte, a.tau)

order = pd.read_csv(a.data / "sample_submission.csv", dtype={"sample_id": "string"})["sample_id"].astype(str).tolist()
s = pd.Series(pred.astype(int), index=tst.sample_id.values)
if not a.smoke:
    assert set(order) == set(tst.sample_id), "sample_submission 과 test.csv ID 불일치"
    sub = pd.DataFrame({"sample_id": order, "label": s.loc[order].values})
    assert len(sub) == 3451 and sub.label.isin(L).all() and sub.sample_id.is_unique
else:
    sub = pd.DataFrame({"sample_id": s.index, "label": s.values})
outp = ROOT / "outputs" / "submissions" / f"{a.name}{'_smoke' if a.smoke else ''}.csv"
outp.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(outp, index=False)
np.save(outp.with_suffix(".proba.npy"), Pte.astype(np.float32))
(outp.with_suffix(".json")).write_text(json.dumps(dict(
    variant=BV, blend_w=BW, models=a.models, tau=a.tau, xgb_device=_xgb_dev["v"],
    val_f1_cap500=float(best.f1_post), val_f1_full=float(R[(R['eval'] == 'full') & (R.model == 'ens') & (R.variant == BV) & (R.blend_w == BW)].f1_post.iloc[0])), indent=1))
log(f"저장: {outp} ({len(sub)}행)")
tt = tst.assign(label_pred=pred).sort_values(["experiment_id", "relative_time_s"], kind="stable")
mono = tt.groupby("experiment_id").label_pred.apply(lambda x: bool((np.diff(x.values) >= 0).all()))
print("제출 라벨 분포:", sub.label.value_counts(normalize=True).sort_index().round(3).to_dict())
print(f"단조 비감소 실험 {int(mono.sum())}/{len(mono)}")
for ref in sorted((ROOT / "outputs" / "submissions").glob("*.csv")):
    if ref != outp and not a.smoke and "smoke" not in ref.name:
        o = pd.read_csv(ref, dtype={"sample_id": "string"}).set_index("sample_id").label
        print(f"  {ref.name} 과 일치율 {(o.loc[sub.sample_id.values].values == sub.label.values).mean():.4f}")
log("DONE")
