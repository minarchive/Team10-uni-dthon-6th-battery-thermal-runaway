"""센서 시계열 단방향(인과) GRU + 누적 순서형 출력. OOF 5-fold(같은 folds.parquet)로 평가하고 LightGBM OOF 와의 블렌드도 본다.

입력(시점마다): 신호 z-점수(가스·압력은 log1p), 실험 첫 관측 대비 변화, 온도 누적최댓값, 전압비, dt, 경과시간 + 정적 조건(tr_method, soc, bat_class, log bat_cc).
미래 정보 없음: 단방향 GRU 와 첫 관측/누적 값만 사용. 학습 시 프레임을 stride 1~3 으로 건너뛴 사본을 무작위로 써서 Test 의 10/15초 간격을 흉내 낸다(건너뛴 행만으로 피처 재계산).
출력: P(단계>=2..6) 5개 sigmoid -> 6-class 확률(cummin 후 인접 차이). 평가는 팀 후처리(확신 cummax tau=0.7)로 LightGBM 과 같은 조건.
"""
import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from common import DATA, ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--drop", type=float, default=0.2)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seeds", default="0")
ap.add_argument("--name", default="gru")
ap.add_argument("--rnn", default="gru", choices=["gru", "lstm"])
ap.add_argument("--tau", type=float, default=0.7)
a = ap.parse_args()
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
log = lambda *m: print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *m, flush=True)
L = list(range(1, 7))
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
SIG = ["meas_v", "meas_t1", "meas_t2", "meas_p", "atmp_t", "thermal_t", "indu_t"] + GAS
tb = table()
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
raw["fold"], raw["label"] = tb.fold.values, tb.label.values
raw["row"] = np.arange(len(raw))
indu_has = raw.groupby("experiment_id").indu_t.transform(lambda s: float(s.notna().any()))
X = raw[SIG].copy()
for c in GAS + ["meas_p"]: X[c] = np.log1p(X[c])
X["indu_t"] = X["indu_t"].fillna(0.0)
lab_mask = (raw.label.values >= 1) & (raw.fold.values >= 0)
mu, sd = X[lab_mask].mean(), X[lab_mask].std() + 1e-6
Z = ((X - mu) / sd).astype(np.float32)
static_cols = pd.DataFrame({"tr": (raw.tr_method == 2).astype(float), "soc": (raw.soc_status == 2).astype(float), "c1": (raw.bat_class == 1).astype(float),
                            "c2": (raw.bat_class == 2).astype(float), "c3": (raw.bat_class == 3).astype(float), "cc": np.log(raw.bat_cc.astype(float)) / 6.0, "indu": indu_has})
S_ = static_cols.values.astype(np.float32)


def seq_feats(idx):
    """idx: 한 실험의 행 번호(시간순). 반환 (T, F) float32. 첫 관측·누적 값만 사용."""
    z = Z.values[idx]
    rel = z - z[0:1]
    t = raw.relative_time_s.values[idx].astype(np.float32)
    dt = np.r_[0.0, np.diff(t)] / 5.0
    t1c = np.maximum.accumulate(Z.values[idx][:, 1]); t2c = np.maximum.accumulate(Z.values[idx][:, 2]); thc = np.maximum.accumulate(Z.values[idx][:, 5])
    v = raw.meas_v.values[idx]; v0 = max(v[0], 1e-6)
    vr = v / v0
    extra = np.stack([t1c, t2c, thc, np.minimum.accumulate(vr), np.clip(dt, 0, 6), t / 1500.0], 1).astype(np.float32)
    return np.concatenate([z, rel, extra, S_[idx]], 1)


exps = {}
for e, g in raw.groupby("experiment_id"):
    ix = g.sort_values("relative_time_s", kind="stable").row.values
    exps[e] = dict(ix=ix, fold=int(raw.fold.values[ix[0]]), lab=raw.label.values[ix[0]] >= 1, split=g.iloc[0].experiment_id and None)
labeled = [e for e, v in exps.items() if v["lab"] and v["fold"] >= 0]
variants = {}
for e in labeled:
    ix = exps[e]["ix"]; vs = [(ix, seq_feats(ix), raw.label.values[ix])]
    for s in (2, 3):
        for o in range(s):
            sub = ix[o::s]
            if len(sub) > 20: vs.append((sub, seq_feats(sub), raw.label.values[sub]))
    variants[e] = vs
nf = variants[labeled[0]][0][1].shape[1]
log(f"실험 {len(labeled)}개 학습용, 입력 차원 {nf}, device {dev}")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, a.hidden), nn.ReLU(), nn.Dropout(a.drop))
        self.gru = (nn.GRU if a.rnn == "gru" else nn.LSTM)(a.hidden, a.hidden, a.layers, batch_first=True, dropout=a.drop if a.layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Dropout(a.drop), nn.Linear(a.hidden, 5))

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h)


def pad(batch):
    T = max(len(b[1]) for b in batch)
    x = np.zeros((len(batch), T, nf), np.float32); y = np.zeros((len(batch), T), np.int64); m = np.zeros((len(batch), T), np.float32)
    for i, (_, f, l) in enumerate(batch):
        x[i, :len(f)] = f; y[i, :len(l)] = l; m[i, :len(l)] = 1
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), torch.from_numpy(m).to(dev)


def to_prob(logits):
    s = torch.cummin(torch.sigmoid(logits), -1).values
    one, zero = torch.ones_like(s[..., :1]), torch.zeros_like(s[..., :1])
    p = (torch.cat([one, s], -1) - torch.cat([s, zero], -1)).clamp_min(1e-6)
    return p / p.sum(-1, keepdim=True)


cnt = np.bincount(raw.label.values[lab_mask].astype(int), minlength=7)[1:].astype(float)
cw = torch.tensor((cnt.sum() / cnt) ** 0.5 / ((cnt.sum() / cnt) ** 0.5).mean(), dtype=torch.float32, device=dev)  # 클래스 가중(제곱근 균형)


def loss_fn(lg, y, m):
    tgt = (y[..., None] > torch.arange(1, 6, device=dev)[None, None]).float()
    w = cw[(y - 1).clamp(0, 5)] * m
    return (nn.functional.binary_cross_entropy_with_logits(lg, tgt, reduction="none").mean(-1) * w).sum() / w.sum()


@torch.no_grad()
def predict(net, es):
    net.eval(); out = {}
    for e in es:
        ix, f, _ = variants[e][0] if e in variants else None
        x = torch.from_numpy(f[None]).to(dev)
        out[e] = (ix, to_prob(net(x))[0].cpu().numpy())
    return out


f1 = lambda y, p: f1_score(y, p, labels=L, average="macro", zero_division=0)
lab_i = np.where(lab_mask)[0]
fr = raw.iloc[lab_i].reset_index(drop=True); y_lab = fr.label.astype(int).values; fold = fr.fold.values
pos = pd.Series(np.arange(len(lab_i)), index=lab_i)
OOFS = []
for seed in [int(s) for s in a.seeds.split(",")]:
    oof = np.zeros((len(y_lab), 6)); rng = np.random.default_rng(seed); torch.manual_seed(seed)
    for k in range(5):
        tr_e = [e for e in labeled if exps[e]["fold"] != k]; te_e = [e for e in labeled if exps[e]["fold"] == k]
        rng.shuffle(tr_e); n_in = max(4, int(0.15 * len(tr_e))); in_e, fit_e = tr_e[:n_in], tr_e[n_in:]
        net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
        best, bstate, bad = -1, None, 0
        for ep in range(a.epochs):
            net.train(); order = rng.permutation(len(fit_e))
            for bi in range(0, len(order), 16):
                batch = [variants[fit_e[i]][rng.integers(len(variants[fit_e[i]]))] for i in order[bi:bi + 16]]
                x, y, m = pad(batch); x = x + 0.02 * torch.randn_like(x)
                opt.zero_grad(); loss_fn(net(x), y, m).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            pr = predict(net, in_e); yy = np.concatenate([raw.label.values[pr[e][0]] for e in in_e]); pp = np.concatenate([pr[e][1].argmax(1) + 1 for e in in_e])
            sc = f1(yy.astype(int), pp)
            if sc > best: best, bad, bstate = sc, 0, {kk: v.clone() for kk, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience: break
        net.load_state_dict(bstate)
        for e, (ix, P) in predict(net, te_e).items(): oof[pos.loc[ix].values] = P
        va = fold == k
        log(f"seed {seed} fold {k}: inner F1 {best:.3f} (ep {ep + 1}) outer F1(후처리) {f1(y_lab[va], F.postprocess(fr[va].reset_index(drop=True), oof[va], a.tau)):.4f}")
    OOFS.append(oof)
O = np.mean(OOFS, 0)
np.save(ROOT / "outputs" / "preds" / f"seq_{a.name}_oof.npy", O.astype(np.float32))
p = F.postprocess(fr, O, a.tau)
lgbm = np.load(ROOT / "outputs" / "preds" / "sensor_lgb_oof.npy"); pl = F.postprocess(fr, lgbm, a.tau)
trm = fr.tr_method.values
print(f"\n== GRU 단독 OOF F1 {f1(y_lab, p):.4f} (argmax {f1(y_lab, O.argmax(1) + 1):.4f}) | 가열 {f1(y_lab[trm == 1], p[trm == 1]):.4f} 과충전 {f1(y_lab[trm == 2], p[trm == 2]):.4f} | 단계별 {np.round(f1_score(y_lab, p, labels=L, average=None), 3).tolist()}")
print(f"   LightGBM(OOF, 같은 fold) {f1(y_lab, pl):.4f}")
bfs = np.array([f1(y_lab[fold == k], pl[fold == k]) for k in range(5)])
for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.7]:
    pb = F.postprocess(fr, (1 - w) * lgbm + w * O, a.tau); fs = np.array([f1(y_lab[fold == k], pb[fold == k]) for k in range(5)])
    print(f"   블렌드 w={w:<3} (GRU 비중) OOF {f1(y_lab, pb):.4f} ({f1(y_lab, pb) - f1(y_lab, pl):+.4f}) 이긴 fold {int((fs > bfs).sum())}/5 | fold차이 {np.round(fs - bfs, 4).tolist()}")
