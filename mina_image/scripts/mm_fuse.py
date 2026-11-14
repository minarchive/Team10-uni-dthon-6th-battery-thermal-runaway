"""Transformer 인코더(센서) + ResNet50(열화상) 초기 융합 구조를 우리 Train+Val 로 처음부터 학습(OOF 5-fold). 외부 학습 가중치는 쓰지 않는다.
센서 분기: 인과적(단방향 마스크) Transformer 인코더, 입력 = 현재까지 최근 W 스텝의 센서 특징(seq_cv 와 같은 47차원 + 유효 플래그). 마지막 토큰 = 현재 프레임.
이미지 분기: timm ResNet50(ImageNet 범용 가중치) stage3 까지 + GAP, 현재 프레임 1장(224 패딩 캐시).
융합: [센서 토큰, 이미지 특징] 이어 붙임 -> MLP -> 6-class. --no_img 1 이면 센서 Transformer 만(이미지 기여 분리용).
평가: folds.parquet 실험 단위 5-fold, 같은 후처리(확신 cummax), LightGBM OOF 와 비교·블렌드. 조기종료는 학습 fold 안쪽 실험 단위 inner split."""
import argparse
import math
import sys
import time
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from common import DATA, ROOT, table

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="mm")
ap.add_argument("--no_img", type=int, default=0)
ap.add_argument("--W", type=int, default=48)
ap.add_argument("--img_dim", type=int, default=0, help=">0 이면 이미지 특징을 이 차원으로 압축(병목)한 뒤 융합")
ap.add_argument("--tab", type=int, default=0, help=">0 이면 LightGBM 인과 피처(F.build_features, 중요도 상위 K개)를 헤드 입력에 이어 붙임. 선택은 라벨 행 전체 중요도(약한 선택 누수)")
ap.add_argument("--soft_a", type=float, default=0.0, help=">0 이면 경계 근처 프레임의 목표를 인접 단계와 섞음(최대 비율 a)")
ap.add_argument("--soft_D", type=float, default=15.0, help="소프트 라벨이 적용되는 경계 거리(초)")
ap.add_argument("--heads", default="none", choices=["none", "tr", "cls", "cell"], help="공유 몸통 위에 그룹별 출력층(tr=방식 2개, cls=형태 3개, cell=형태x방식 6개)")
ap.add_argument("--mod_drop", type=float, default=0.0, help="학습 중 이미지 특징을 이 확률로 0 으로(모달리티 드롭아웃)")
ap.add_argument("--stage2", type=int, default=0, help=">0 이면 2단계 학습: 1단계는 이미지 없이(센서·표만) --epochs, 2단계는 이미지 가지를 0 초기화 잔차로 붙여 이 에폭 동안 작은 lr 로 미세조정")
ap.add_argument("--lr2", type=float, default=0.1, help="2단계에서 센서·표 가지 lr 배율")
ap.add_argument("--gas", default="all", choices=["all", "sum"], help="sum=개별 가스 10종을 0 으로, 가스합만 meas_co2 칸에 유지(OOF +0.009)")
ap.add_argument("--rules", type=int, default=0, help="1=방식별 규칙 신호(thermal-atmp, T2-T1, thermal 기울기 등의 누적 극값, 이벤트 이후 경과)를 과충전/가열 분리 복사본으로 헤드 입력에 추가")
ap.add_argument("--wcell", type=int, default=0, help="1=샘플링 확률을 (bat_class x 단계) 칸별 역빈도^power 로(파우치 소수 단계 보강)")
ap.add_argument("--minmax", type=int, default=0, help="1=센서를 학습 라벨 행 기준 Min-Max(0~1)로 정규화")
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--frac", type=float, default=0.3)
ap.add_argument("--power", type=float, default=0.5)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--lr_img", type=float, default=1e-4)
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, str(ROOT / "ts_model"))
import features as F

dev = "cuda"
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
T0 = time.time()
log = lambda *m: print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *m, flush=True)
L = list(range(1, 7))
GAS = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
SIG = ["meas_v", "meas_t1", "meas_t2", "meas_p", "atmp_t", "thermal_t", "indu_t"] + GAS

tb = table()
N = len(tb)
raw = pd.concat([pd.read_csv(DATA / s / f"{s}.csv", dtype={"sample_id": "string", "experiment_id": "string"}) for s in ["train", "validation", "test"]], ignore_index=True)
raw["sample_id"], raw["experiment_id"] = raw.sample_id.astype(str), raw.experiment_id.astype(str)
raw = raw.set_index("sample_id", drop=False).loc[tb.sample_id.astype(str).values].reset_index(drop=True)
y_all = tb.label.values.astype(int) - 1          # 라벨 없음(test) = -2
fold_all = tb.fold.values
lab = (tb.label.values >= 1) & (fold_all >= 0)
te_idx = np.where(tb.label.values == -1)[0]
if a.gas == "sum":
    _G = ["meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl", "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2"]
    _s = raw[_G].sum(axis=1)
    for _g in _G: raw[_g] = 0.0
    raw["meas_co2"] = _s
indu_has = raw.groupby("experiment_id").indu_t.transform(lambda s: float(s.notna().any()))
X = raw[SIG].copy()
for c in GAS + ["meas_p"]:
    X[c] = np.log1p(X[c])
X["indu_t"] = X["indu_t"].fillna(0.0)
mu, sd = X[lab].mean(), X[lab].std() + 1e-6
if a.minmax:
    mn, mx = X[lab].min(), X[lab].max()
    Z = ((X - mn) / (mx - mn + 1e-6)).clip(-0.5, 1.5).astype(np.float32).values
else:
    Z = ((X - mu) / sd).astype(np.float32).values
S_ = pd.DataFrame({"tr": (raw.tr_method == 2).astype(float), "soc": (raw.soc_status == 2).astype(float), "c1": (raw.bat_class == 1).astype(float),
                   "c2": (raw.bat_class == 2).astype(float), "c3": (raw.bat_class == 3).astype(float), "cc": np.log(raw.bat_cc.astype(float)) / 6.0, "indu": indu_has}).values.astype(np.float32)
t_all = raw.relative_time_s.values.astype(float)
eid = raw.experiment_id.values
FEAT = np.zeros((N, 47), np.float32)
WIN = np.full((N, a.W), -1, np.int64)
for e, ix in pd.Series(np.arange(N)).groupby(eid).indices.items():
    ix = ix[np.argsort(t_all[ix], kind="stable")]
    z = Z[ix]; rel = z - z[0:1]; t = t_all[ix].astype(np.float32); dt = np.r_[0.0, np.diff(t)] / 5.0
    t1c = np.maximum.accumulate(z[:, 1]); t2c = np.maximum.accumulate(z[:, 2]); thc = np.maximum.accumulate(z[:, 5])
    v = raw.meas_v.values[ix]; vr = v / max(v[0], 1e-6)
    extra = np.stack([t1c, t2c, thc, np.minimum.accumulate(vr), np.clip(dt, 0, 6), t / 1500.0], 1).astype(np.float32)
    FEAT[ix] = np.concatenate([z, rel, extra, S_[ix]], 1)
    for p, r in enumerate(ix):                       # 현재 프레임을 마지막 칸, 과거 프레임을 왼쪽으로(부족하면 -1)
        lo = max(0, p - a.W + 1); seg = ix[lo:p + 1]; WIN[r, a.W - len(seg):] = seg
TABg = None
RUL = None
if a.rules:
    GAS_ = [c for c in raw.columns if c.startswith("meas_") and c not in ("meas_v", "meas_a", "meas_t1", "meas_t2", "meas_p", "meas_t1_plus_t2")]
    ser = {"thermal-amb": (raw.thermal_t - raw.atmp_t).values, "thermal": raw.thermal_t.values, "t2-t1": (raw.meas_t2 - raw.meas_t1).values, "thermal-t1": (raw.thermal_t - raw.meas_t1).values,
           "t1-amb": (raw.meas_t1 - raw.atmp_t).values, "v/v0": (raw.meas_v / raw.groupby("experiment_id").meas_v.transform("first").replace(0, np.nan)).values,
           "p-p0": (raw.meas_p - raw.groupby("experiment_id").meas_p.transform("first")).values, "gas_sum": np.log1p(raw[GAS_].sum(axis=1)).values}
    cols_, names_ = [], []
    for nm, x in ser.items():
        cx, cn = np.full(N, np.nan), np.full(N, np.nan)
        for e, ix in pd.Series(np.arange(N)).groupby(eid).indices.items():
            ix = ix[np.argsort(t_all[ix], kind="stable")]; xs = pd.Series(x[ix]); cx[ix] = xs.cummax().values; cn[ix] = xs.cummin().values
        cols_ += [cx, cn]; names_ += [nm + "_cummax", nm + "_cummin"]
    for c in ("meas_t1", "thermal_t"):                                   # 30초 기울기 최댓값
        x = raw[c].values.astype(float); sl = np.full(N, np.nan)
        for e, ix in pd.Series(np.arange(N)).groupby(eid).indices.items():
            ix = ix[np.argsort(t_all[ix], kind="stable")]; tt = t_all[ix]; j = np.searchsorted(tt, tt - 30, side="right") - 1; ok = j >= 0; o = np.full(len(ix), np.nan); o[ok] = (x[ix][ok] - x[ix][j[ok]]) / np.maximum(tt[ok] - tt[j[ok]], 1); sl[ix] = pd.Series(o).cummax().values
        cols_.append(sl); names_.append(c + "_slope30_cummax")
    # 인과적: T1 누적최댓값이 45°C 이상이 된 '이후' 행에만 경과 시간을 주고, 이전 행은 0 + 도달 지시자 채널(미래 도달 시각을 쓰지 않음)
    t1m = raw.meas_t1.groupby(raw.experiment_id).cummax().values; reached = (t1m >= 45)
    t45 = pd.Series(np.where(reached, t_all, np.nan)).groupby(raw.experiment_id.values).transform("min").values   # 도달 이후 행에서만 사용(그 행에서 t45 는 과거)
    cols_.append(np.where(reached, t_all - t45, 0.0)); names_.append("since_T1_45")
    cols_.append(reached.astype(np.float32)); names_.append("reached_T1_45")
    R = np.stack(cols_, 1).astype(np.float32); li_ = np.where(lab)[0]
    mu_r, sd_r = np.nanmean(R[li_], 0), np.nanstd(R[li_], 0) + 1e-6
    R = np.nan_to_num((R - mu_r) / sd_r, nan=0.0).clip(-6, 6)
    oc = (raw.tr_method.values == 2).astype(np.float32)[:, None]
    RUL = np.concatenate([R * oc, R * (1 - oc)], 1).astype(np.float32)       # 과충전 전용 / 가열 전용 복사본
    print("규칙 신호 입력:", RUL.shape[1], "개", flush=True)
if a.tab:
    import lightgbm as lgb
    XT = F.build_features(raw)                                   # 같은 인과 피처 735개(행 순서 = tb 순서)
    li = np.where(lab)[0]
    m0 = F.build_model(); m0.set_params(n_jobs=6, n_estimators=150, learning_rate=0.08)
    m0.fit(XT.iloc[li], tb.label.values[li].astype(int))
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:a.tab]
    XA = XT.iloc[:, top].astype(np.float32).values
    mu_t, sd_t = np.nanmean(XA[li], 0), np.nanstd(XA[li], 0) + 1e-6
    XA = np.nan_to_num((XA - mu_t) / sd_t, nan=0.0).clip(-6, 6).astype(np.float32)
    if RUL is not None: XA = np.concatenate([XA, RUL], 1)
    TABg = torch.from_numpy(XA).to(dev)
    print("LightGBM 피처 입력:", a.tab, "개 (상위:", list(XT.columns[top[:5]]), ")", flush=True)
if a.tab == 0 and RUL is not None:
    TABg = torch.from_numpy(RUL).to(dev)
_grp = {"none": np.zeros(N, np.int64), "tr": (raw.tr_method.values == 2).astype(np.int64), "cls": raw.bat_class.values.astype(np.int64) - 1,
        "cell": (raw.bat_class.values.astype(np.int64) - 1) * 2 + (raw.tr_method.values == 2).astype(np.int64)}[a.heads]
NH = int(_grp.max()) + 1
GRPg = torch.from_numpy(_grp).to(dev)
_dist = np.full(N, np.inf, np.float32); _adj = np.where(y_all >= 0, y_all, 0).astype(np.int64)
if a.soft_a > 0:      # 학습 라벨에서만 계산(라벨 있는 행). 경계까지의 거리(초)와 경계 건너편 단계
    for _e, _ix in pd.Series(np.arange(N)).groupby(eid).indices.items():
        _ix = _ix[np.argsort(t_all[_ix], kind="stable")]; _yy = y_all[_ix]
        if (_yy < 0).any(): continue
        _tt = t_all[_ix]; _ch = np.where(np.diff(_yy) != 0)[0] + 1
        for _c in _ch:
            for _j in range(len(_ix)):
                _d = min(abs(_tt[_j] - _tt[_c]), abs(_tt[_j] - _tt[_c - 1]))
                if _d < _dist[_ix[_j]]: _dist[_ix[_j]] = _d; _adj[_ix[_j]] = _yy[_c] if _j < _c else _yy[_c - 1]
DISTg = torch.from_numpy(_dist).to(dev); ADJg = torch.from_numpy(_adj).to(dev)
FEATg = torch.from_numpy(FEAT).to(dev)
WINg = torch.from_numpy(WIN).to(dev)
nf = FEAT.shape[1] + 1
imgs = np.load(ROOT / "data" / "img224_pad.npy", mmap_mode="r") if not a.no_img else None
log(f"N={N} 라벨 {int(lab.sum())} test {len(te_idx)} | 입력 {nf} | 창 {a.W} | 이미지 {'사용' if not a.no_img else '미사용'}")


def load_imgs(idx):
    idx = np.asarray(idx); o = np.argsort(idx)
    arr = np.empty((len(idx), 224, 224, 3), np.uint8); arr[o] = imgs[idx[o]]
    return torch.from_numpy(arr).to(dev, non_blocking=True)


class MM(nn.Module):
    def __init__(s):
        super().__init__()
        d = a.d
        s.inp = nn.Linear(nf, d)
        s.pos = nn.Parameter(torch.zeros(1, a.W, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True), a.layers)
        s.register_buffer("cmask", torch.triu(torch.ones(a.W, a.W, dtype=torch.bool), 1))
        hin = d + (a.tab if a.tab else 0) + (RUL.shape[1] if RUL is not None else 0)
        if not a.no_img:
            m = timm.create_model("resnet50.a1_in1k", pretrained=True)
            s.register_buffer("mean", torch.tensor(m.pretrained_cfg["mean"]).view(1, 3, 1, 1)); s.register_buffer("std", torch.tensor(m.pretrained_cfg["std"]).view(1, 3, 1, 1))
            s.img = nn.Sequential(m.conv1, m.bn1, m.act1, m.maxpool, m.layer1, m.layer2, m.layer3)
            if a.img_dim:
                s.proj = nn.Sequential(nn.Linear(1024, a.img_dim), nn.LayerNorm(a.img_dim))
                hin += a.img_dim
            else:
                hin += 1024
            s.fdim = a.img_dim or 1024
        s.heads = nn.ModuleList([nn.Sequential(nn.Dropout(0.3), nn.Linear(hin, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 6)) for _ in range(NH)])

    def forward(s, win, im=None):
        valid = (win >= 0).float()
        x = FEATg[win.clamp(min=0)] * valid[..., None]
        x = torch.cat([x, valid[..., None]], -1)
        h = s.enc(s.inp(x) + s.pos, mask=s.cmask)[:, -1]
        if TABg is not None: h = torch.cat([h, TABg[win[:, -1]]], 1)
        if not a.no_img:
          if im is None:
            f = torch.zeros(h.shape[0], s.fdim, device=h.device, dtype=h.dtype)
          else:
            f = s.img(((im.permute(0, 3, 1, 2).float() / 255 - s.mean) / s.std).contiguous(memory_format=torch.channels_last)).mean((2, 3))
            if a.img_dim: f = s.proj(f)
            if s.training and a.mod_drop > 0: f = f * (torch.rand(f.shape[0], 1, device=f.device) > a.mod_drop).float()
          h = torch.cat([h, f.to(h.dtype)], 1)
        if NH == 1: return s.heads[0](h)
        out = torch.stack([hd(h) for hd in s.heads], 0)                       # (NH,B,6)
        return out[GRPg[win[:, -1]], torch.arange(h.shape[0], device=h.device)]


IMG_ON = [not a.stage2]


def batch(idx):
    return WINg[torch.as_tensor(idx, device=dev)], (load_imgs(idx) if (not a.no_img and IMG_ON[0]) else None)


@torch.no_grad()
def predict(net, idx, bs=256):
    net.eval(); out = []
    for i in range(0, len(idx), bs):
        w, im = batch(idx[i:i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(net(w, im).float(), 1).cpu())
    return torch.cat(out).numpy()


mf1 = lambda yy, p: f1_score(yy, p, labels=list(range(6)), average="macro", zero_division=0)
if a.smoke:
    a.epochs, a.folds = 1, "0"
    keep = np.sort(rng.choice(np.where(lab)[0], 3000, replace=False)); lab = np.zeros(N, bool); lab[keep] = True
fr_all = tb.assign(experiment_id=tb.experiment_id.astype(str))
pos = np.cumsum(lab) - 1
oof = np.zeros((int(lab.sum()), 6), np.float32)
tsum, nd = np.zeros((len(te_idx), 6)), 0
covered = np.zeros(N, bool)
for f in [int(x) for x in a.folds.split(",")]:
    va = np.where(lab & (fold_all == f))[0]; tr = np.where(lab & (fold_all != f))[0]
    gi, gm = next(GroupShuffleSplit(1, test_size=0.15, random_state=a.seed).split(tr, groups=tb.experiment_id.values[tr]))
    itr, imon = tr[gi], tr[gm]; imon = imon[rng.permutation(len(imon))[:4000]]
    net = MM().to(dev); IMG_ON[0] = not a.stage2
    cnt = np.bincount(y_all[itr], minlength=6); pw = (1.0 / np.maximum(cnt, 1)) ** a.power; p = pw[y_all[itr]]
    if a.wcell:
        ck = raw.bat_class.values[itr].astype(int) * 10 + y_all[itr]; cc_ = pd.Series(ck).map(pd.Series(ck).value_counts()).values.astype(float); p = cc_ ** (-a.power)
    p = p / p.sum()
    n_ep = max(1, int(a.frac * len(itr))); spe = math.ceil(n_ep / a.bs)
    best, bstate = -1, None

    def run_phase(n_epochs, lr_mult_oth, tag):
        global best, bstate
        img_p = [q for n, q in net.named_parameters() if n.startswith("img.")]; oth = [q for n, q in net.named_parameters() if not n.startswith("img.")]
        opt = torch.optim.AdamW([dict(params=oth, lr=a.lr * lr_mult_oth), dict(params=img_p, lr=a.lr_img)], weight_decay=0.05)
        base = [g["lr"] for g in opt.param_groups]
        total = spe * n_epochs; step = 0; bad = 0
        for ep in range(n_epochs):
            net.train(); sel = itr[rng.choice(len(itr), n_ep, replace=True, p=p)]; tl = 0.0; t0 = time.time()
            for i in range(spe):
                b = sel[i * a.bs:(i + 1) * a.bs]
                if len(b) < 2: continue
                sc = min(1.0, (step + 1) / max(1, spe // 2)) * 0.5 * (1 + math.cos(math.pi * step / total))
                for g, bl in zip(opt.param_groups, base): g["lr"] = bl * sc
                w, im = batch(b); yb = torch.from_numpy(y_all[b]).long().to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = net(w, im)
                if a.soft_a > 0:
                    bb = torch.as_tensor(b, device=dev); aa = a.soft_a * (1 - DISTg[bb] / a.soft_D).clamp(0, 1)
                    tgt = (1 - aa)[:, None] * Fn.one_hot(yb, 6).float() + aa[:, None] * Fn.one_hot(ADJg[bb], 6).float(); tgt = tgt * 0.95 + 0.05 / 6
                    loss = -(tgt * Fn.log_softmax(lg.float(), 1)).sum(1).mean()
                else:
                    loss = Fn.cross_entropy(lg.float(), yb, label_smoothing=0.05)
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += loss.item(); step += 1
            pm = predict(net, imon); fm = mf1(y_all[imon], pm.argmax(1))
            log(f"fold {f} {tag} ep {ep + 1}/{n_epochs} loss {tl / spe:.3f} mon_f1 {fm:.3f} ({time.time() - t0:.0f}s)")
            if fm > best: best, bad, bstate = fm, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience: break

    run_phase(a.epochs, 1.0, "P1(센서·표)" if a.stage2 else "")
    if a.stage2:
        net.load_state_dict(bstate); IMG_ON[0] = True
        with torch.no_grad():
            for hd in net.heads: hd[1].weight[:, -net.fdim:] = 0.0          # 이미지 열 0 초기화 -> 시작 함수 = 1단계 모델
        bstate = {k: v.detach().clone() for k, v in net.state_dict().items()}
        log(f"fold {f} 2단계 시작: 1단계 모니터 F1 {best:.3f}, 이미지 가지 잔차 {a.stage2}에폭, 센서·표 lr x{a.lr2}")
        run_phase(a.stage2, a.lr2, "P2(+이미지)")
    net.load_state_dict(bstate)
    pv = predict(net, va); oof[pos[va]] = pv; covered[va] = True
    tsum += predict(net, te_idx); nd += 1
    fv = fr_all.iloc[va].reset_index(drop=True)
    log(f"== fold {f}: outer F1 argmax {mf1(y_all[va], pv.argmax(1)):.4f} | 후처리 {f1_score(y_all[va] + 1, F.postprocess(fv, pv, 0.7), labels=L, average='macro'):.4f}")
np.save(ROOT / "outputs" / "preds" / f"ft_mm_{a.name}_oof.npy", oof)
if not a.smoke:
    np.save(ROOT / "outputs" / "preds" / f"ft_mm_{a.name}_test.npy", (tsum / max(nd, 1)).astype(np.float32))
    fr = tb[lab].reset_index(drop=True); fr["experiment_id"] = fr.experiment_id.astype(str); y = fr.label.astype(int).values; fold = fr.fold.values
    so = np.load(ROOT / "outputs/preds/sensor_lgb_oof.npy"); pe, ps = F.postprocess(fr, oof, 0.7), F.postprocess(fr, so, 0.7)
    f1 = lambda p: f1_score(y, p, labels=L, average="macro", zero_division=0)
    fs = lambda p: np.array([f1_score(y[fold == k], p[fold == k], labels=L, average="macro") for k in range(5)])
    t1all = raw.meas_t1.values.astype(float); cmx = np.zeros(N)
    for _e, _ix in pd.Series(np.arange(N)).groupby(eid).indices.items():
        _ix = _ix[np.argsort(t_all[_ix], kind="stable")]; cmx[_ix] = np.maximum.accumulate(t1all[_ix])
    rch = (cmx >= 45)[lab]
    def gate(Pm):
        G = Pm.copy(); G[~rch] = 0; G[~rch, 0] = 1.0; G[rch, 0] = 0; sm = G[rch].sum(1, keepdims=True); G[rch] = np.where(sm > 0, G[rch] / np.where(sm > 0, sm, 1), 0.2); G[rch, 0] = 0; return G
    pg = F.postprocess(fr, gate(oof), 0.7)
    print(f"\n== {a.name} 규칙 후처리(L1 규칙) 적용: 단독 OOF {f1(pg):.4f} ({f1(pg) - f1(pe):+.4f}) | 단계별 {np.round(f1_score(y, pg, labels=L, average=None), 3).tolist()}")
    for w in (0.3,):
        pbl = F.postprocess(fr, (1 - w) * so + w * gate(oof), 0.7); print(f"   규칙 후처리 후 LightGBM 블렌드 w={w}: {f1(pbl):.4f} ({f1(pbl) - f1(ps):+.4f})")
    np.save(ROOT / "outputs" / "preds" / f"ft_mm_{a.name}_oof_gated.npy", gate(oof).astype(np.float32))
    print(f"\n== {a.name} 단독 OOF F1 {f1(pe):.4f} (argmax {f1(oof.argmax(1) + 1):.4f}) | LightGBM {f1(ps):.4f} | 단계별 {np.round(f1_score(y, pe, labels=L, average=None), 3).tolist()}")
    bf = fs(ps)
    for w in (0.2, 0.3, 0.5):
        pb = F.postprocess(fr, (1 - w) * so + w * oof, 0.7); print(f"   블렌드 w={w}: {f1(pb):.4f} ({f1(pb) - f1(ps):+.4f}) 이긴 fold {int((fs(pb) > bf).sum())}/5")
