"""실험 클러스터 MoE (시계열 전용, LightGBM).

아이디어
- 학습하면서 '비슷하게 진행되는 실험'끼리 K개 그룹으로 묶고, 그룹마다 전문가(expert)를 둔다.
- 추론 시에는 게이트가 현재까지의 센서 피처만 보고 그룹 확률을 내고, 전문가 출력을 그 확률로 섞는다.

구성 (외부 fold마다, 그 fold의 학습 실험만 사용)
1. base      : 전체 학습 행으로 LightGBM (v3 설정과 같음, 500 trees, balanced 가중치)
2. 초기 그룹 : 실험별 진행 프로파일(단계별 지속시간, 전환 시점의 t1·전압)로 KMeans(K)
3. hard-EM   : 내부 GroupKFold(3)로 '그 실험을 안 본' base + 그룹 전문가를 만들고,
               실험마다 평균 log-likelihood가 가장 높은 전문가의 그룹으로 재배정 (EM_ROUNDS회)
               실험 MIN_SIZE개 미만 그룹은 없애고 구성원을 다음 순위 그룹으로 보냄
4. 전문가    : base에서 이어서(init_model) 그룹 행으로 CONT_ROUNDS 트리 추가 학습
5. 게이트    : 행 단위 LightGBM 다중분류 (입력 = 같은 735 인과 피처, 정답 = 실험의 그룹)
6. 출력      : p = Σ_k gate_k · expert_k

K=1 은 'base + 전체 행으로 트리 추가'로, 트리 수 증가 효과만 보는 대조군이다.
K 선택은 내부 CV 기준(log-likelihood)으로 하고(외부 점수를 보지 않음), 외부 CV 점수도 함께 보고한다.
'oracle' 은 평가 실험을 자기 라벨로 가장 잘 맞는 그룹에 배정한 상한선(진단용, 제출 불가)이다.

평가: exp_v4.py와 같은 조건. GroupKFold(5, shuffle, seed 0) × GBM 시드, 500행 캡, 확신 cummax τ=0.7, Macro F1.
모든 피처는 인과적(현재 시점 이하 관측만). 그룹 라벨은 학습 실험의 라벨로만 만든다.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import moe_gate as MG
from exp_img import F, load, log

LAB = F.LABELS
D = Path('/home/nxcolelxu/unid/data/battery_data_light')
EM_ROUNDS = 3
MIN_SIZE = 4
CONT_ROUNDS = 150
EPS = 1e-6

BASE_P = dict(objective='multiclass', num_class=6, learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.3, lambda_l2=1.0, verbose=-1, num_threads=int(__import__('os').environ.get('CMOE_THREADS', '0')))
CONT_P = dict(BASE_P, num_leaves=15)
GATE_P = dict(objective='multiclass', learning_rate=0.05, num_leaves=15, min_data_in_leaf=100,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.3, lambda_l2=1.0, verbose=-1, num_threads=int(__import__('os').environ.get('CMOE_THREADS', '0')))


def class_w(y):
    """sklearn class_weight='balanced'와 같은 클래스 가중치 (y는 0..5), 길이 6."""
    cnt = np.bincount(y, minlength=6).astype(float)
    return len(y) / (6 * np.maximum(cnt, 1))


class LGBBackend:
    name = 'lgbm_cpu'

    def base(self, X, y, seed):
        return lgb.train(dict(BASE_P, seed=seed), lgb.Dataset(X, y, weight=class_w(y)[y], free_raw_data=False), num_boost_round=500)

    def cont(self, base, X, y, wc, seed):
        return lgb.train(dict(CONT_P, seed=seed), lgb.Dataset(X, y, weight=wc[y], free_raw_data=False),
                         num_boost_round=CONT_ROUNDS, init_model=base, keep_training_booster=False)

    def gate(self, X, g, K, seed):
        return lgb.train(dict(GATE_P, num_class=K, seed=seed), lgb.Dataset(X, g), num_boost_round=200)

    def predict(self, m, X):
        return m.predict(X)


class XGBBackend:
    """XGBoost GPU(hist). LightGBM 설정에 최대한 맞춤 (exp_img.XGBBalanced와 같은 값)."""
    name = 'xgb_cuda'

    def _p(self, seed, num_class, leaves=31, eta=0.03, mcw=50):
        return dict(objective='multi:softprob', num_class=num_class, eta=eta, max_leaves=leaves, max_depth=0,
                    grow_policy='lossguide', min_child_weight=mcw, subsample=0.8, colsample_bytree=0.3,
                    reg_lambda=1.0, tree_method='hist', device='cuda', seed=seed, verbosity=0, nthread=8)

    def base(self, X, y, seed):
        return xgb.train(self._p(seed, 6), xgb.DMatrix(X, y, weight=class_w(y)[y]), 500)

    def cont(self, base, X, y, wc, seed):
        return xgb.train(self._p(seed, 6, leaves=15), xgb.DMatrix(X, y, weight=wc[y]), CONT_ROUNDS, xgb_model=base)

    def gate(self, X, g, K, seed):
        return xgb.train(self._p(seed, K, leaves=15, eta=0.05, mcw=100), xgb.DMatrix(X, g), 200)

    def predict(self, m, X):
        return m.predict(xgb.DMatrix(X))


BK = LGBBackend()


# ------------------------------------------------------------------ 실험 프로파일 (학습 실험 라벨 사용)
def profiles(frame):
    rows = {}
    for e, g in frame.sort_values('relative_time_s').groupby('experiment_id'):
        t, l = g.relative_time_s.to_numpy(float), g.label.to_numpy(int)
        t1 = np.fmax.accumulate(np.nan_to_num(g.meas_t1.to_numpy(float), nan=-1e9))
        v = g.meas_v.to_numpy(float); v0 = v[0] if v[0] > 0 else np.nan
        r = {}
        for s in (2, 3, 4, 5):
            m = l == s
            r[f'logdur{s}'] = np.log1p(t[m].max() - t[m].min() + 5) if m.any() else 0.0
        for s in (3, 4, 5, 6):
            m = l >= s
            r[f't1at{s}'] = t1[np.argmax(m)] if m.any() else t1[-1]
            r[f'vr_at{s}'] = (v[np.argmax(m)] / v0) if m.any() else v[-1] / v0
        r['log_t45'] = np.log1p(t[np.argmax(l >= 2)])
        rows[e] = r
    P = pd.DataFrame.from_dict(rows, orient='index')
    P = P.fillna(P.median())
    return P


def kmeans_groups(P, exps, K, seed):
    Z = P.loc[exps]
    Z = ((Z - Z.mean()) / Z.std().replace(0, 1)).fillna(0).to_numpy()
    return dict(zip(exps, KMeans(K, n_init=10, random_state=seed).fit_predict(Z)))


# ------------------------------------------------------------------ 로그우도
def exp_loglik(P, y, eids):
    """실험별 평균 log p(y). 반환 dict e -> float."""
    lp = np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))
    s = pd.Series(lp).groupby(np.asarray(eids)).mean()
    return s.to_dict()


def prune(groups, ll, K):
    """MIN_SIZE 미만 그룹 제거, 구성원은 남은 그룹 중 log-lik 최대로. 그룹 번호 0..K'-1로 재매핑."""
    alive = set(range(K))
    while True:
        sizes = pd.Series(groups).value_counts()
        small = [k for k in alive if sizes.get(k, 0) < MIN_SIZE]
        if not small or len(alive) == 1:
            break
        k_rm = min(small, key=lambda k: sizes.get(k, 0))
        alive.discard(k_rm)
        for e, k in groups.items():
            if k == k_rm:
                groups[e] = max(alive, key=lambda j: ll[e][j]) if ll is not None else min(alive)
    remap = {k: i for i, k in enumerate(sorted(alive))}
    return {e: remap[k] for e, k in groups.items()}, len(alive)


# ------------------------------------------------------------------ 한 외부 fold
class FoldCtx:
    """외부 fold의 학습 데이터와 K에 무관하게 공유되는 것들(base, 내부 fold base)."""

    def __init__(self, Xtr, ytr, eids, seed):
        self.X, self.y, self.e, self.seed = Xtr, ytr, np.asarray(eids), seed
        self.exps = np.array(sorted(set(self.e)))
        self.wc = class_w(ytr)
        self.base = BK.base(Xtr, ytr, seed)
        self.inner = []
        for a_e, b_e in GroupKFold(3, shuffle=True, random_state=seed).split(self.exps, groups=self.exps):
            ea, eb = set(self.exps[a_e]), set(self.exps[b_e])
            ma, mb = np.isin(self.e, list(ea)), np.isin(self.e, list(eb))
            ib = BK.base(Xtr[ma], ytr[ma], seed)
            self.inner.append((ma, mb, ib, BK.predict(ib, Xtr[mb])))


def em_groups(ctx, P_prof, K):
    """hard-EM으로 실험 그룹을 학습. 반환: groups(dict), K', 내부 CV 기준(MoE oracle ll, base ll)."""
    groups = kmeans_groups(P_prof, list(ctx.exps), K, ctx.seed) if K > 1 else {e: 0 for e in ctx.exps}
    groups, K = prune(groups, None, K)
    hist = []
    for r in range(EM_ROUNDS if K > 1 else 1):
        ll = {e: np.full(K, -np.inf) for e in ctx.exps}
        base_ll = {}
        for ma, mb, ib, pb in ctx.inner:
            ga = np.array([groups[e] for e in ctx.e[ma]])
            yb, eb = ctx.y[mb], ctx.e[mb]
            base_ll.update(exp_loglik(pb, yb, eb))
            Xa, ya = ctx.X[ma], ctx.y[ma]
            for k in range(K):
                sel = ga == k
                if sel.sum() == 0:
                    continue
                ex = BK.cont(ib, Xa[sel], ya[sel], ctx.wc, ctx.seed)
                for e, v in exp_loglik(BK.predict(ex, ctx.X[mb]), yb, eb).items():
                    ll[e][k] = v
        new = {e: int(np.argmax(ll[e])) for e in ctx.exps}
        crit = float(np.mean([ll[e].max() for e in ctx.exps]))
        changed = sum(new[e] != groups[e] for e in ctx.exps)
        hist.append(dict(round=r, crit_oracle_ll=crit, base_ll=float(np.mean(list(base_ll.values()))), changed=changed))
        if K == 1:
            break
        groups, K = prune(new, ll, K)
        if changed == 0:
            break
    return groups, K, hist


def fit_moe(ctx, groups, K):
    g_rows = np.array([groups[e] for e in ctx.e])
    experts = [BK.cont(ctx.base, ctx.X[g_rows == k], ctx.y[g_rows == k], ctx.wc, ctx.seed) for k in range(K)]
    gate = None
    if K > 1:
        gate = BK.gate(ctx.X, g_rows, K, ctx.seed)
    return experts, gate


def predict_moe(experts, gate, X):
    Pk = np.stack([BK.predict(ex, X) for ex in experts], 1)      # n × K × 6
    G = BK.predict(gate, X) if gate is not None else np.ones((len(X), 1))
    return (G[:, :, None] * Pk).sum(1), Pk, G


# ------------------------------------------------------------------ 메인
def post_f1(frame, P):
    fc, l = labels(frame, P)
    y = fc.label.to_numpy(int)
    return f1_score(y, l, labels=LAB, average='macro'), f1_score(y, l, labels=LAB, average=None)


def labels(frame, P):
    c = F.cap_rows(frame).index.to_numpy()
    fc = frame.iloc[c].reset_index(drop=True)
    return fc, F.postprocess(fc, P[c], 0.7)


def run_part(tr, X, y, eid, meta, P_prof, splits, seed, fi, Ks, tag, t0):
    """외부 fold 하나 × 시드 하나. 결과를 parts/에 저장."""
    a, b = splits[fi]
    ctx = FoldCtx(X[a], y[a], eid[a], seed)
    out = {'idx': b, 'v3': BK.predict(ctx.base, X[b])}
    info = {}
    log(f'seed {seed} fold {fi} base 완료 [{time.time() - t0:.0f}s]')
    for K in Ks:
        groups, K2, hist = em_groups(ctx, P_prof, K)
        experts, gate = fit_moe(ctx, groups, K2)
        Pm, Pk, G = predict_moe(experts, gate, X[b])
        out[f'K{K}'] = Pm
        out[f'K{K}_hard'] = Pk[np.arange(len(b)), G.argmax(1)]
        eb = eid[b]
        ll = np.stack([pd.Series(np.log(np.clip(Pk[np.arange(len(b)), k, y[b]], EPS, 1))).groupby(eb).mean() for k in range(K2)], 1)
        best = dict(zip(sorted(set(eb)), ll.argmax(1)))
        out[f'K{K}_oracle'] = Pk[np.arange(len(b)), [best[e] for e in eb]]
        gm = pd.Series(groups)
        purity = gm.groupby(meta.loc[gm.index].values).agg(lambda s_: s_.value_counts().iloc[0] / len(s_)).mean()
        gate_acc = float(np.mean(G.argmax(1) == np.array([best[e] for e in eb]))) if K2 > 1 else 1.0
        info[K] = dict(K_eff=K2, sizes=gm.value_counts().sort_index().tolist(), meta_purity=float(purity),
                       gate_acc_vs_oracle=gate_acc, em=hist, groups={e: int(k) for e, k in groups.items()})
        log(f'seed {seed} fold {fi} K={K}→{K2} sizes {info[K]["sizes"]} purity {purity:.2f} gate≈oracle {gate_acc:.2f} '
            f'innerLL oracle {hist[-1]["crit_oracle_ll"]:.4f} base {hist[-1]["base_ll"]:.4f} [{time.time() - t0:.0f}s]')
    Path('parts').mkdir(exist_ok=True)
    np.savez(f'parts/cmoe_{tag}_s{seed}_f{fi}.npz', **out)
    Path(f'parts/cmoe_{tag}_s{seed}_f{fi}.json').write_text(json.dumps(info, default=float))


def merge(tr, y, tag):
    """parts를 모아 시드별 CV 점수와 요약 출력."""
    files = sorted(Path('parts').glob(f'cmoe_{tag}_s*_f*.npz'))
    by_seed = {}
    for f in files:
        s_, f_ = f.stem.split('_s')[-1].split('_f')
        by_seed.setdefault(int(s_), {})[int(f_)] = f
    res = {}
    for seed, fs in sorted(by_seed.items()):
        if len(fs) < 5:
            log(f'seed {seed}: fold {sorted(fs)}만 있음 → 건너뜀'); continue
        oof = {}
        for f in fs.values():
            z = np.load(f)
            for k in z.files:
                if k == 'idx':
                    continue
                oof.setdefault(k, np.zeros((len(tr), 6)))[z['idx']] = z[k]
        for name, P in oof.items():
            fc, l = labels(tr, P); yy = fc.label.to_numpy(int)
            r = res.setdefault(name, dict(cv=[], per_class=[]))
            r['cv'].append(f1_score(yy, l, labels=LAB, average='macro'))
            r['per_class'].append(f1_score(yy, l, labels=LAB, average=None).round(4).tolist())
        np.savez(f'cluster_moe_{tag}_seed{seed}.npz', train_id=tr.sample_id.to_numpy(str), **oof)
    log(f'\n=== {tag} 요약 (시드 {sorted(by_seed)}) ===')
    base = np.array(res['v3']['cv'])
    for name, r in res.items():
        d = np.array(r['cv']) - base
        log(f'  {name:12s} CV {np.mean(r["cv"]):.4f} ± {np.std(r["cv"]):.4f}  Δv3 {d.mean():+.4f} (시드별 {np.round(d, 4).tolist()})  '
            f'per-class {np.mean(r["per_class"], 0).round(3).tolist()}')
    Path(f'res_cluster_moe_{tag}.json').write_text(json.dumps(res, indent=1, default=float))


def main():
    global BK
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ks', default='1,2,3,4,6')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--tag', default='lgb')
    ap.add_argument('--backend', default='lgbm_cpu', choices=['lgbm_cpu', 'xgb_cuda'])
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    BK = XGBBackend() if args.backend == 'xgb_cuda' else LGBBackend()
    t0 = time.time()
    tr = load(D, 'train').reset_index(drop=True)
    y = tr.label.to_numpy(int) - 1
    if args.merge:
        return merge(tr, y, args.tag)
    X = F.build_features(tr).to_numpy(np.float32)
    eid = tr.experiment_id.astype(str).to_numpy()
    meta = tr.groupby('experiment_id')[['bat_class', 'bat_cc', 'soc_status', 'tr_method']].first().astype(str).agg('|'.join, axis=1)
    P_prof = profiles(tr)
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))
    log(f'backend {BK.name}, 피처 {X.shape}, 준비 {time.time() - t0:.0f}s')
    Ks = [int(k) for k in args.Ks.split(',')]
    for seed in [int(s_) for s_ in args.seeds.split(',')]:
        for fi in [int(f_) for f_ in args.folds.split(',')]:
            run_part(tr, X, y, eid, meta, P_prof, splits, seed, fi, Ks, args.tag, t0)


if __name__ == '__main__':
    main()
