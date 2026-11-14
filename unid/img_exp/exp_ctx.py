"""시계열 전용 GBM에 맥락 피처 3종을 단독 / 2개 / 3개 조합으로 추가했을 때 효과 비교.

  A traj : 궤적 맥락 — 기울기의 변화(가속도), 짧은/긴 기울기 차, 최대 승온 속도와 그 이후 경과 시간,
           승온 속도 임계 최초 도달 이후 경과 시간. (v3에 이미 있는 기울기·온도 임계 경과 시간은 중복 추가 안 함)
  B rel  : 실험 기준 상대값 — 실험 처음 60초 평균(60초 전에는 지금까지의 평균) 대비 Δ, 누적최대−기준, 상대 변화
  C proto: 프로토타입 거리 + kNN 이웃 단계 분포. fold 학습 실험만으로 공간(log 압축·표준화·PCA 32) 구성.
           학습 행은 자기 실험을 뺀 프로토타입/이웃(leave-one-experiment-out), 평가 행은 학습 실험 전체가 풀.

평가는 exp_v4.py와 같다: Train GroupKFold(5, seed 0) × GBM 시드 {0,1,2}, 500행 캡, 확신 cummax τ=0.7.
Val: Train 전체 학습, 3시드 평균, 1회 채점. 실험 단위 bootstrap Δ (vs v3).
GBM은 GBM_BACKEND (xgb_cuda 권장), kNN·PCA는 torch GPU.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import os

import moe_gate as M
import exp_img
from exp_img import F, load, log

TAG = os.environ.get('CTX_TAG', '')       # 출력 파일 접미사 (GPU/CPU 중복 실행 시 덮어쓰기 방지)
if os.environ.get('GBM_BACKEND') == 'xgb_cpu':
    def lgbm(seed):                        # XGBoost CPU hist, 설정은 xgb_cuda 래퍼와 동일
        m = exp_img.XGBBalanced(seed)
        m.m.set_params(device='cpu', n_jobs=int(os.environ.get('GBM_THREADS', '12')))
        return m
elif os.environ.get('GBM_BACKEND', 'lgbm_cpu') == 'lgbm_cpu' and 'GBM_THREADS' in os.environ:
    def lgbm(seed):                        # LightGBM CPU, 스레드 수 제한 (로그인 노드 공유 시)
        return exp_img.lgbm(seed).set_params(n_jobs=int(os.environ['GBM_THREADS']))
else:
    lgbm = exp_img.lgbm

LAB = F.LABELS
D = Path('/home/nxcolelxu/unid/data/battery_data_light')
SEEDS = tuple(int(s) for s in os.environ.get('CTX_SEEDS', '0,1,2').split(','))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAJ_SIG = ['meas_t1', 'meas_t2', 't_max', 'meas_p', 'thermal_t', 'gas_sum', 'meas_v']
RATE_SIG = ['meas_t1', 't_max']
RATE_LV = (0.05, 0.1, 0.2, 0.5)          # °C/s
REL_SIG = ['meas_v', 'meas_t1', 'meas_t2', 't_max', 'meas_p', 'atmp_t', 'thermal_t', 'indu_t'] + F.GAS + ['gas_sum']
BASE_S = 60
PCA_K, KNN_K = 32, 25


# ---------------------------------------------------------------------------
# 실험별 시간순 인덱스 도우미
# ---------------------------------------------------------------------------
def exp_groups(frame):
    eid = frame.experiment_id.astype(str).to_numpy()
    t = pd.to_numeric(frame.relative_time_s).to_numpy(float)
    return t, [idx[np.argsort(t[idx], kind='mergesort')] for idx in pd.Series(eid).groupby(eid, sort=False).indices.values()]


def lag(t, groups, v, w):
    out = np.full(len(v), np.nan)
    for idx in groups:
        tt = t[idx]
        j = np.searchsorted(tt, tt - w, side='right') - 1
        ok = j >= 0
        out[idx[ok]] = v[idx[j[ok]]]
    return out


def since_first(t, groups, hit):
    out = np.full(len(hit), np.nan)
    for idx in groups:
        h = hit[idx]
        if h.any():
            f = int(np.argmax(h))
            out[idx[f:]] = t[idx[f:]] - t[idx[f]]
    return out


def cummax_and_since(t, groups, v):
    cm, since = np.full(len(v), np.nan), np.full(len(v), np.nan)
    for idx in groups:
        x = v[idx]
        c = np.fmax.accumulate(np.where(np.isfinite(x), x, -np.inf))
        peak_t = np.where(np.isfinite(x) & (x >= c), t[idx], np.nan)
        peak_t = pd.Series(peak_t).ffill().to_numpy()
        cm[idx] = np.where(np.isfinite(c), c, np.nan)
        since[idx] = t[idx] - peak_t
    return cm, since


# ---------------------------------------------------------------------------
# A. 궤적 맥락
# ---------------------------------------------------------------------------
def feat_traj(frame, X):
    t, g = exp_groups(frame)
    cols = {}
    for s in TRAJ_SIG:
        for w in (30, 60, 120):
            sl = X[f'{s}_slope{w}'].to_numpy(float)
            cols[f'tr_{s}_acc{w}'] = sl - lag(t, g, sl, w)            # 기울기의 변화 (2차 차분)
        s30, s120 = X[f'{s}_slope30'].to_numpy(float), X[f'{s}_slope120'].to_numpy(float)
        cols[f'tr_{s}_sl30m120'] = s30 - s120                          # 최근 꺾임
        cm, since = cummax_and_since(t, g, s30)
        cols[f'tr_{s}_slope30_cmax'] = cm
        cols[f'tr_{s}_since_maxslope'] = since
    for s in RATE_SIG:
        r = X[f'{s}_slope30'].to_numpy(float)
        for lv in RATE_LV:
            cols[f'tr_{s}_since_rate{lv}'] = since_first(t, g, np.nan_to_num(r, nan=-np.inf) >= lv)
    return pd.DataFrame(cols, index=X.index).astype(np.float32)


# ---------------------------------------------------------------------------
# B. 실험 기준 상대값
# ---------------------------------------------------------------------------
def feat_rel(frame, X):
    d = F._prep(frame)
    t, g = exp_groups(frame)
    cols = {}
    for s in REL_SIG:
        x_all = d[s].to_numpy(float)
        base = np.full(len(x_all), np.nan)
        for idx in g:
            x = x_all[idx]
            m = (t[idx] <= BASE_S) & np.isfinite(x)
            cs, c = np.cumsum(np.where(m, x, 0.0)), np.cumsum(m)
            base[idx] = np.where(c > 0, cs / np.maximum(c, 1), np.nan)   # 60초 이후엔 고정
        delta = x_all - base
        cm = X[f'{s}_cummax'].to_numpy(float)
        cols[f'rel_{s}_d'] = delta
        cols[f'rel_{s}_cmax_d'] = cm - base
        cols[f'rel_{s}_frac'] = delta / (np.abs(base) + 1.0)
    xv = d['meas_v'].to_numpy(float)
    vb = xv - cols['rel_meas_v_d']
    cols['rel_meas_v_ratio'] = np.where(vb > 0, xv / np.where(vb > 0, vb, 1), np.nan)
    return pd.DataFrame(cols, index=X.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


# ---------------------------------------------------------------------------
# C. 프로토타입 거리 + kNN (fold 의존)
# ---------------------------------------------------------------------------
class Space:
    """학습 행으로 fit: 부호 보존 log → 표준화(±5 클립, NaN→0) → PCA k."""

    def __init__(self, X, k=PCA_K):
        Z = np.sign(X) * np.log1p(np.abs(X))
        self.mu = np.nan_to_num(np.nanmean(Z, 0)); self.sd = np.nan_to_num(np.nanstd(Z, 0), nan=1.0) + 1e-6
        Zt = torch.from_numpy(self._z(X)).to(DEV)
        self.c = Zt.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(Zt - self.c, q=k, center=False, niter=4)
        self.V = V[:, :k]

    def _z(self, X):
        Z = np.sign(X) * np.log1p(np.abs(X))
        return np.nan_to_num(np.clip((Z - self.mu) / self.sd, -5, 5), nan=0.0).astype(np.float32)

    def __call__(self, X):
        return (torch.from_numpy(self._z(X)).to(DEV) - self.c) @ self.V


def feat_proto_knn(Xp, y, exp, bat, Xq, exp_q, bat_q):
    """Xp: 풀(학습 행) 피처, Xq: 질의 행. 질의 행과 같은 실험의 풀 행은 프로토타입·이웃 모두에서 제외."""
    sp = Space(Xp)
    P, Q = sp(Xp), sp(Xq)
    ue = {e: i for i, e in enumerate(np.unique(np.concatenate([exp, exp_q])))}
    ep = torch.tensor([ue[e] for e in exp], device=DEV)
    eq = torch.tensor([ue[e] for e in exp_q], device=DEV)
    yk = torch.tensor(y - 1, device=DEV)
    ub = {b: i for i, b in enumerate(np.unique(np.concatenate([bat, bat_q])))}
    bp = torch.tensor([ub[b] for b in bat], device=DEV)
    bq = torch.tensor([ub[b] for b in bat_q], device=DEV)
    nE, nB, d = len(ue), len(ub), P.shape[1]
    # 프로토타입 합계: (bat, stage), 전역 stage, (exp, stage)
    S = torch.zeros(nB, 6, d, device=DEV).index_put_((bp, yk), P, accumulate=True)
    C = torch.zeros(nB, 6, device=DEV).index_put_((bp, yk), torch.ones(len(P), device=DEV), accumulate=True)
    SE = torch.zeros(nE, 6, d, device=DEV).index_put_((ep, yk), P, accumulate=True)
    CE = torch.zeros(nE, 6, device=DEV).index_put_((ep, yk), torch.ones(len(P), device=DEV), accumulate=True)
    SG, CG = S.sum(0), C.sum(0)
    out = np.zeros((len(Q), 17), np.float32)
    ks = torch.arange(1, 7, device=DEV, dtype=torch.float32)
    for j in range(0, len(Q), 2048):
        q, e, b = Q[j:j + 2048], eq[j:j + 2048], bq[j:j + 2048]
        sb, cb = S[b] - SE[e], C[b] - CE[e]                         # 자기 실험 제외 (질의가 풀에 없으면 SE=0)
        sg, cg = SG[None] - SE[e], CG[None] - CE[e]
        proto = torch.where((cb > 0)[..., None], sb / cb.clamp(min=1)[..., None], sg / cg.clamp(min=1)[..., None])
        dist = (q[:, None, :] - proto).norm(dim=2)
        dist = torch.where((cb > 0) | (cg > 0), dist, torch.full_like(dist, float('nan')))
        dfill = torch.nan_to_num(dist, nan=1e6)
        w = torch.softmax(-dfill, 1)
        out[j:j + 2048, 0:6] = dist.cpu().numpy()
        out[j:j + 2048, 6] = (dist[:, 2] - dist[:, 1]).cpu().numpy()          # d(L3) − d(L2)
        out[j:j + 2048, 7] = (dfill.argmin(1) + 1).float().cpu().numpy()
        out[j:j + 2048, 8] = (w * ks).sum(1).cpu().numpy()
        # kNN
        dd = torch.cdist(q, P)
        dd = dd.masked_fill(e[:, None] == ep[None, :], float('inf'))
        dk, ik = dd.topk(KNN_K, dim=1, largest=False)
        lab = yk[ik]
        frac = torch.stack([(lab == k).float().mean(1) for k in range(6)], 1)
        out[j:j + 2048, 9:15] = frac.cpu().numpy()
        out[j:j + 2048, 15] = (lab.float() + 1).mean(1).cpu().numpy()
        out[j:j + 2048, 16] = dk.mean(1).cpu().numpy()
    return out


C_COLS = [f'pr_d{k}' for k in range(1, 7)] + ['pr_d3m2', 'pr_argmin', 'pr_soft_stage'] + \
         [f'knn_f{k}' for k in range(1, 7)] + ['knn_stage', 'knn_dist']


# ---------------------------------------------------------------------------
def capped_labels(frame, P):
    c = F.cap_rows(frame).index.to_numpy()
    fc = frame.iloc[c].reset_index(drop=True)
    return fc, F.postprocess(fc, P[c], 0.7)


def main():
    tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
    ytr = tr.label.astype(int).to_numpy()
    base, A, B = {}, {}, {}
    for nm, fr in (('tr', tr), ('va', va), ('te', te)):
        X = F.build_features(fr)
        base[nm], A[nm], B[nm] = X, feat_traj(fr, X), feat_rel(fr, X)
    log(f'features: v3 {base["tr"].shape[1]}, A traj {A["tr"].shape[1]}, B rel {B["tr"].shape[1]}, C proto/knn {len(C_COLS)}  (device {DEV})')

    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    Xb = base['tr'].to_numpy(np.float32)
    exp_tr, bat_tr = tr.experiment_id.astype(str).to_numpy(), tr.bat_class.astype(str).to_numpy()
    C_fold = []
    for a, b in splits:                                   # fold별: 학습 행은 LOEO, 평가 행은 학습 실험 풀
        Cf = np.zeros((len(tr), len(C_COLS)), np.float32)
        Cf[a] = feat_proto_knn(Xb[a], ytr[a], exp_tr[a], bat_tr[a], Xb[a], exp_tr[a], bat_tr[a])
        Cf[b] = feat_proto_knn(Xb[a], ytr[a], exp_tr[a], bat_tr[a], Xb[b], exp_tr[b], bat_tr[b])
        C_fold.append(pd.DataFrame(Cf, columns=C_COLS, index=base['tr'].index))
    C_full = {'tr': pd.DataFrame(feat_proto_knn(Xb, ytr, exp_tr, bat_tr, Xb, exp_tr, bat_tr), columns=C_COLS, index=base['tr'].index)}
    for nm, fr in (('va', va), ('te', te)):
        C_full[nm] = pd.DataFrame(feat_proto_knn(Xb, ytr, exp_tr, bat_tr, base[nm].to_numpy(np.float32),
                                                 fr.experiment_id.astype(str).to_numpy(), fr.bat_class.astype(str).to_numpy()),
                                  columns=C_COLS, index=base[nm].index)
    log('proto/knn features done')

    names = ['A', 'B', 'C']
    combos = [()] + [c for r in (1, 2, 3) for c in itertools.combinations(names, r)]
    if len(sys.argv) > 1:
        combos = [() if a == 'v3' else tuple(a.split('+')) for a in sys.argv[1:]]

    def design(nm, combo, Cdf):
        parts = [base[nm]] + [A[nm]] * ('A' in combo) + [B[nm]] * ('B' in combo) + [Cdf] * ('C' in combo)
        return pd.concat(parts, axis=1)

    res, cache, save = {}, {}, {}
    for combo in combos:
        name = 'v3' + ''.join('+' + c for c in combo)
        cv, oof_avg = [], np.zeros((len(tr), 6))
        Xfold = [design('tr', combo, C_fold[k]) for k in range(5)]
        for s in SEEDS:
            oof = np.zeros((len(tr), 6))
            for k, (a, b) in enumerate(splits):
                oof[b] = lgbm(s).fit(Xfold[k].iloc[a], ytr[a]).predict_proba(Xfold[k].iloc[b])
            fc, l = capped_labels(tr, oof)
            cv.append(f1_score(fc.label.to_numpy(int), l, labels=LAB, average='macro'))
            oof_avg += oof / len(SEEDS)
        Xtr_full, Xva, Xte = design('tr', combo, C_full['tr']), design('va', combo, C_full['va']), design('te', combo, C_full['te'])
        pv, pt = np.zeros((len(va), 6)), np.zeros((len(te), 6))
        for s in SEEDS:
            m = lgbm(s).fit(Xtr_full, ytr)
            pv += m.predict_proba(Xva) / len(SEEDS); pt += m.predict_proba(Xte) / len(SEEDS)
        fct, lt = capped_labels(tr, oof_avg)
        fcv, lv = capped_labels(va, pv)
        cache[name] = (fct, lt, fcv, lv)
        res[name] = dict(n_feat=int(Xtr_full.shape[1]), cv_per_seed=cv, cv_mean=float(np.mean(cv)), cv_std=float(np.std(cv)),
                         cv_per_class=f1_score(fct.label.to_numpy(int), lt, labels=LAB, average=None).round(3).tolist(),
                         val=float(f1_score(fcv.label.to_numpy(int), lv, labels=LAB, average='macro')),
                         val_per_class=f1_score(fcv.label.to_numpy(int), lv, labels=LAB, average=None).round(3).tolist())
        r = res[name]
        line = (f'{name:9s} feats {r["n_feat"]:4d}  CV {r["cv_mean"]:.4f} ± {r["cv_std"]:.4f} {r["cv_per_class"]} | '
                f'Val {r["val"]:.4f} {r["val_per_class"]}')
        if name != 'v3':
            _, lt0, _, lv0 = cache['v3']
            cb = M.boot_diff(fct, lt0, lt, np.arange(len(fct)), n=1000)
            vb = M.boot_diff(fcv, lv0, lv, np.arange(len(fcv)), n=1000)
            r['cv_boot'], r['val_boot'] = cb, vb
            line += f'\n    Δ vs v3: CV {cb[0]:+.4f} [{cb[1]:+.4f},{cb[2]:+.4f}] P>0 {cb[3]:.2f} | Val {vb[0]:+.4f} [{vb[1]:+.4f},{vb[2]:+.4f}] P>0 {vb[3]:.2f}'
        log(line)
        save.update({f'{name}_oof': oof_avg, f'{name}_val': pv, f'{name}_test': pt})
        Path(f'res_ctx{TAG}.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez(f'ctx_probs{TAG}.npz', train_id=tr.sample_id.to_numpy(str), val_id=va.sample_id.to_numpy(str),
             test_id=te.sample_id.to_numpy(str), **save)
    log('saved res_ctx.json, ctx_probs.npz')


if __name__ == '__main__':
    main()
