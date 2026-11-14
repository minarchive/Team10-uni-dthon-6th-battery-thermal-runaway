"""궤적 유사도 기반 사례 추론(kNN). 예측 대상 프레임 t 에서 '지금까지의 궤적'을 학습 실험들의 같은 길이 궤적과 비교해, 가장 비슷한 실험들의 라벨 타임라인으로 단계를 예측한다.

인과성: 프레임 t 의 예측은 같은 실험의 현재·과거 관측만 쓴다. 학습 실험의 라벨 타임라인은 학습 정보(허용).
정렬: 각 실험의 T1 누적최대가 처음 45°C 이상이 된 시각(L1->L2 규칙)을 0 으로 맞춘 '사건 시간' s = t - t45 (과거 관측으로 결정 가능). s<0 이면 L1 로 확정.
특징 곡선(s 에 대해 5초 간격으로 리샘플): T1 상승량(첫 관측 대비), thermal_t 상승량, 압력 상승량, 전압비, 가스합(log1p), T2-T1.
거리: 후보 학습 실험의 [0, s] 구간 곡선과 현재 실험의 [0, s] 곡선 사이 정규화 유클리드 거리(최근 W초 가중). 같은 tr_method 실험끼리만 비교(옵션).
예측: 거리가 가까운 k 개 학습 실험의 시각 s 에서의 라벨을 거리 가중 투표(확률). 이후 팀 후처리(확신 cummax)를 적용해 센서 LightGBM OOF 와 같은 조건으로 비교.
평가: 실험 단위 5-fold(folds.parquet), 같은 fold 에서 학습 fold 실험만 이웃 후보."""
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import DATA, ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=7)
ap.add_argument("--W", type=float, default=300.0, help="최근 W초 구간에 가중(그 이전은 가중 0.3)")
ap.add_argument("--same_tr", type=int, default=1, help="1=같은 tr_method 실험만 이웃")
ap.add_argument("--tau", type=float, default=0.7)
a = ap.parse_args()
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F

L = list(range(1, 7))
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
lab = (tb.label.values >= 1) & (tb.fold.values >= 0)
fr = tb[lab].reset_index(drop=True)
fr["experiment_id"] = fr.experiment_id.astype(str)
raw = raw.set_index("sample_id").loc[fr.sample_id.astype(str).values].reset_index()          # fr 행 순서
raw["fold"] = fr.fold.values
y = fr.label.astype(int).values
fold = fr.fold.values
eid = raw.experiment_id.values
t_all = raw.relative_time_s.values.astype(float)
raw["gas"] = np.log1p(raw[GAS].sum(axis=1))
CURVES = ["t1r", "thr", "pr", "vr", "gas", "t21"]


def curves(df):
    """한 실험(시간순)의 특징 곡선 (T, 6): 첫 관측 대비 상승량/비율. 현재·과거만 사용."""
    t1 = df.meas_t1.values; v = df.meas_v.values
    return np.stack([t1 - t1[0], df.thermal_t.values - df.thermal_t.values[0], df.meas_p.values - df.meas_p.values[0],
                     v / max(v[0], 1e-6), df.gas.values, df.meas_t2.values - t1], 1)


E = {}
for e, ix in pd.Series(np.arange(len(raw))).groupby(eid).indices.items():
    ix = ix[np.argsort(t_all[ix], kind="stable")]
    d = raw.iloc[ix]
    cm = np.maximum.accumulate(d.meas_t1.values)
    j45 = int(np.argmax(cm >= 45)) if (cm >= 45).any() else len(ix)
    E[e] = dict(ix=ix, t=t_all[ix], C=curves(d), j45=j45, t45=t_all[ix][j45] if j45 < len(ix) else np.inf, y=y[ix], tr=int(d.tr_method.iloc[0]), fold=int(d.fold.iloc[0]))
scale = np.nanstd(np.concatenate([v["C"] for v in E.values()]), axis=0) + 1e-6
GRID = 5.0


def resample(ev, s_max):
    """사건 시간 s=0..s_max (5초 격자)에서의 곡선 (과거 관측만; 격자 시각 이하의 마지막 관측값 사용)."""
    s = np.arange(0, s_max + 1e-9, GRID) + ev["t45"]
    j = np.searchsorted(ev["t"], s, side="right") - 1
    j = np.clip(j, 0, len(ev["t"]) - 1)
    return ev["C"][j] / scale


def label_at(ev, s):
    tt = ev["t45"] + s
    j = np.clip(np.searchsorted(ev["t"], tt, side="right") - 1, 0, len(ev["t"]) - 1)
    return ev["y"][j]


P = np.zeros((len(y), 6))
exps = list(E.keys())
for k in range(5):
    train_e = [e for e in exps if E[e]["fold"] != k]
    for e in [e for e in exps if E[e]["fold"] == k]:
        ev = E[e]
        cand = [c for c in train_e if (not a.same_tr) or E[c]["tr"] == ev["tr"]]
        for jj in range(len(ev["ix"])):
            row = ev["ix"][jj]
            if jj < ev["j45"]:      # T1 45°C 미도달 -> L1
                P[row, 0] = 1.0; continue
            s = ev["t"][jj] - ev["t45"]
            n = int(s // GRID) + 1
            q = ev["C"][np.clip(np.searchsorted(ev["t"], ev["t45"] + np.arange(n) * GRID, side="right") - 1, 0, len(ev["t"]) - 1)] / scale
            w = np.where((s - np.arange(n) * GRID) <= a.W, 1.0, 0.3)[:, None]
            dist = []
            for c in cand:
                r = resample(E[c], s)[:n]
                dist.append(np.sqrt((w * (q - r) ** 2).sum() / w.sum()))
            dist = np.array(dist); o = np.argsort(dist)[:a.k]
            wt = 1.0 / (dist[o] + 1e-3)
            for oi, ww in zip(o, wt):
                P[row, label_at(E[cand[oi]], s) - 1] += ww
        ix = ev["ix"]; Pe = P[ix]; sm = Pe.sum(1, keepdims=True); P[ix] = Pe / np.where(sm > 0, sm, 1)
    print(f"fold {k} 완료", flush=True)
np.save(ROOT / "outputs" / "preds" / f"traj_knn_k{a.k}_oof.npy", P.astype(np.float32))
f1 = lambda yy, p: f1_score(yy, p, labels=L, average="macro", zero_division=0)
pk = F.postprocess(fr, P, a.tau); lg = np.load(ROOT / "outputs" / "preds" / "sensor_lgb_oof.npy"); pl = F.postprocess(fr, lg, a.tau)
print(f"\n궤적 kNN(k={a.k}, W={a.W}, 같은 tr={a.same_tr}) OOF F1 후처리 {f1(y, pk):.4f} (argmax {f1(y, P.argmax(1) + 1):.4f}) | 센서 LightGBM {f1(y, pl):.4f}")
print("단계별 kNN", np.round(f1_score(y, pk, labels=L, average=None), 3).tolist(), "| 센서", np.round(f1_score(y, pl, labels=L, average=None), 3).tolist())
fs = lambda p: np.array([f1(y[fold == kk], p[fold == kk]) for kk in range(5)])
bf = fs(pl)
for w in (0.2, 0.3, 0.5):
    pb = F.postprocess(fr, (1 - w) * lg + w * P, a.tau); b = fs(pb)
    print(f"  블렌드 kNN 비중 {w}: OOF {f1(y, pb):.4f} ({f1(y, pb) - f1(y, pl):+.4f}) 이긴 fold {int((b > bf).sum())}/5")
