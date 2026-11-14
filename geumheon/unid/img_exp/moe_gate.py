"""경계별 순서형 게이트 (센서 전문가 + 이미지 전문가).

L2/L3 경계(k=3)에서만 이미지를 섞는다. 나머지 경계는 센서 그대로.

    logit P'(y≥3) = a·S3 + b + g(ctx)·(β·I3 + γ)
      S3  = 센서 logit P(≥3)
      I3  = 이미지 logit P(≥3)의 과거 60초 평균 (프레임 단위 잡음 완화)
      g   = σ(w·[1, 1−conf, log1p(t_since/30), is_pred3, ΔI 60s])
      conf = 센서 최대확률, t_since = 센서 argmax가 현재 단계에 들어온 뒤 경과 초 (인과적)
    적용 범위: 센서 argmax ∈ {2,3} 인 프레임만. 결과 P'(≥3)는 [P(≥4), P(≥2)] 로 잘라 누적확률 일관성 유지.
    변형 'up' : P'(≥3) = max(P(≥3), 게이트 출력) — 올리기만 (단조 후처리에서 잘못 올리면 되돌릴 수 없으므로)

선택 규칙: 전부 Train OOF(GroupKFold 5, seed 0)에서 중첩 CV로 점수 → 1위 설정만 Validation에서 한 번 평가.
비교군: 센서 단독, 수작업 규칙(센서 L2 & conf<c & 이미지 P(≥3)>θ → L3, (c,θ)는 학습 fold에서 선택).

    python moe_gate.py --images ft_convnext_tiny.npz ft_efficientnet_b0.npz [ft_ord_dinov2s_u4.npz ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402

LAB = F.LABELS
EPS = 1e-5
VARIANTS = {'g3': (3,), 'g5': (5,), 'g35': (3, 5)}
RULE_GRID = [(c, t) for c in (0.7, 0.8, 0.9, 0.95, 0.99) for t in (0.6, 0.7, 0.8, 0.9)]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_split(data_dir, split):
    df = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'},
                     usecols=lambda c: c in ('sample_id', 'experiment_id', 'relative_time_s', 'label'))
    df['sample_id'] = df.sample_id.astype(str)
    return df.reset_index(drop=True)


def cum_of(P):
    """클래스 확률 n×6 → P(≥k), k=2..6 (n×5)."""
    return np.cumsum(P[:, ::-1], 1)[:, ::-1][:, 1:]


def probs_of(C):
    C = np.minimum.accumulate(np.clip(C, 0, 1), axis=1)
    full = np.concatenate([np.ones((len(C), 1)), C, np.zeros((len(C), 1))], 1)
    return np.clip(full[:, :-1] - full[:, 1:], 0, 1)


def causal_ctx(frame, Ps, i3):
    """실험별 시간순으로 인과적 컨텍스트 피처 계산."""
    n = len(frame)
    t_all = frame.relative_time_s.to_numpy(float)
    i3m = np.empty(n); di = np.empty(n); tsince = np.empty(n)
    pred = LAB[Ps.argmax(1)]
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        t, x = t_all[idx], i3[idx]
        cs = np.concatenate([[0], np.cumsum(x)])
        lo = np.searchsorted(t, t - 60, side='left')
        hi = np.arange(len(t)) + 1
        m = (cs[hi] - cs[lo]) / (hi - lo)
        i3m[idx] = m
        j = np.clip(np.searchsorted(t, t - 60, side='right') - 1, 0, None)
        di[idx] = m - m[j]
        p = pred[idx]
        start = np.zeros(len(p))
        for r in range(1, len(p)):
            start[r] = start[r - 1] if p[r] == p[r - 1] else t[r]
        start[0] = t[0]
        tsince[idx] = t - start
    return i3m, di, tsince


def build(frame, Ps, Ci, k=3):
    """경계 k (P(y≥k)) 하나에 대한 게이트 입력. Ci: 이미지 P(≥2..6) n×5."""
    c = k - 2
    i_k = Ci[:, c]
    i_m, di, tsince = causal_ctx(frame, Ps, i_k)
    Cs = cum_of(Ps)
    pred = LAB[Ps.argmax(1)]
    return dict(
        k=k, S3=logit(Cs[:, c]), I3=logit(i_m), i3=i_k, i3m=i_m, Cs=Cs, pred=pred, conf=Ps.max(1),
        G=np.stack([np.ones(len(frame)), 1 - Ps.max(1), np.log1p(tsince / 30), (pred == k).astype(float), di * 4], 1),
        region=np.isin(pred, [k - 1, k]),
    )


class Gate(torch.nn.Module):
    def __init__(self, gdim):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.0)); self.b = torch.nn.Parameter(torch.tensor(0.0))
        self.beta = torch.nn.Parameter(torch.tensor(0.5)); self.gam = torch.nn.Parameter(torch.tensor(0.0))
        self.w = torch.nn.Parameter(torch.zeros(gdim))

    def forward(self, S, I, G):
        g = torch.sigmoid(G @ self.w)
        return self.a * S + self.b + g * (self.beta * I + self.gam), g


def fit_gate(X, y, l2=1e-2):
    r = X['region']
    S, I, G = (torch.tensor(v[r], dtype=torch.float64) for v in (X['S3'], X['I3'], X['G']))
    t = torch.tensor((y[r] >= X['k']).astype(float))
    m = Gate(G.shape[1]).double()
    opt = torch.optim.LBFGS(m.parameters(), max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z, _ = m(S, I, G)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, t) + l2 * (m.w.pow(2).sum() + m.beta ** 2)
        loss.backward()
        return loss
    opt.step(closure)
    return m


def apply_gate(m, X, up=False, C=None):
    """X['k'] 경계의 누적확률을 게이트 출력으로 교체. C를 넘기면 그 위에 이어서 적용(여러 경계)."""
    C = (X['Cs'] if C is None else C).copy()
    c, r = X['k'] - 2, X['region']
    with torch.no_grad():
        z, g = m(*(torch.tensor(v[r], dtype=torch.float64) for v in (X['S3'], X['I3'], X['G'])))
    p = 1 / (1 + np.exp(-z.numpy()))
    if up:
        p = np.maximum(p, C[r, c])
    hi = C[r, c - 1] if c > 0 else np.ones(r.sum())
    lo = C[r, c + 1] if c < 4 else np.zeros(r.sum())
    C[r, c] = np.clip(p, lo, hi)
    return C


def gate_probs(models, Xs, up=False):
    C = None
    for m, X in zip(models, Xs):
        C = apply_gate(m, X, up=up, C=C)
    return probs_of(C)


def rule_labels(frame, Ps, X, c, th):
    lab = F.postprocess(frame, Ps, 0.7)
    hit = (lab == 2) & (X['conf'] < c) & (X['i3'] > th)
    lab = np.where(hit, 3, lab)
    t = frame.relative_time_s.to_numpy(float)
    out = lab.copy()
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        out[idx] = np.maximum.accumulate(lab[idx])
    return out


def mf1(y, p):
    return f1_score(y, p, labels=LAB, average='macro')


def score(frame, labels, cap_idx):
    y = frame.label.to_numpy(int)[cap_idx]
    return mf1(y, labels[cap_idx])


def boot_diff(frame, la, lb, cap_idx, n=2000, seed=0):
    """실험 단위 bootstrap: macro F1(lb) − macro F1(la) 의 분포."""
    rng = np.random.default_rng(seed)
    sub = frame.iloc[cap_idx].reset_index(drop=True)
    y, a, b = sub.label.to_numpy(int), la[cap_idx], lb[cap_idx]
    groups = list(sub.groupby('experiment_id', sort=False).indices.values())
    d = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d.append(mf1(y[idx], b[idx]) - mf1(y[idx], a[idx]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def error_report(frame, lab, Pi_cum, cap_idx, name):
    sub = frame.iloc[cap_idx].reset_index(drop=True)
    y, p = sub.label.to_numpy(int), lab[cap_idx]
    out = {}
    for (t, q) in [(3, 2), (5, 4), (2, 3), (5, 6)]:
        m = (y == t) & (p == q)
        per = sub[m].groupby('experiment_id').size().sort_values(ascending=False)
        out[f'L{t}->L{q}'] = dict(n=int(m.sum()), n_exp=int(len(per)), top3_share=float(per.head(3).sum() / max(m.sum(), 1)),
                                  per_exp=per.head(6).astype(int).tolist())
    # 센서가 L4로 예측한 프레임에서 이미지 P(≥5)가 실제 L5를 가르는가
    m4 = p == 4
    if Pi_cum is not None and (y[m4] == 5).any() and (y[m4] == 4).any():
        pi5 = Pi_cum[cap_idx, 3][m4]
        out['auc_img_ge5_on_pred4'] = float(roc_auc_score(y[m4] == 5, pi5))
        e5 = sub[m4 & (y == 5)].experiment_id
        out['pred4_true5_n_exp'] = int(e5.nunique())
    print(f'[{name}] errors:', json.dumps(out, ensure_ascii=False))
    return out


def load_img(path, ids):
    z = np.load(path)
    s = pd.Series(np.arange(len(z['sample_id'])), index=z['sample_id'].astype(str))
    ix = s.loc[ids].to_numpy()
    C = z['cum'][ix] if 'cum' in z.files else cum_of(z['probs'][ix])
    return C.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, default=Path('/home/nxcolelxu/unid/data/battery_data_light'))
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--out', type=Path, default=Path('res_moe_gate.json'))
    ap.add_argument('--show_all_val', action='store_true', help='CV 1위가 아닌 설정의 Val도 출력 (참고용, 선택에 쓰지 말 것)')
    ap.add_argument('--write_test', type=Path, default=None, help='선택된 설정으로 Test 제출 파일 작성')
    args = ap.parse_args()

    tr, va, te = (load_split(args.data_dir, s) for s in ('train', 'validation', 'test'))
    zo = np.load('seq_base_oof.npz'); zv = np.load('seq_base_valtest.npz')
    Ps_tr = pd.DataFrame(zo['probs'], index=zo['sample_id'].astype(str)).loc[tr.sample_id].to_numpy(float)
    Ps_va = pd.DataFrame(zv['val_probs'], index=zv['val_sample_id'].astype(str)).loc[va.sample_id].to_numpy(float)
    Ps_te = pd.DataFrame(zv['test_probs'], index=zv['test_sample_id'].astype(str)).loc[te.sample_id].to_numpy(float)
    y_tr = tr.label.to_numpy(int)
    cap_tr = F.cap_rows(tr).index.to_numpy()
    cap_va = F.cap_rows(va).index.to_numpy()
    folds = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=tr.experiment_id))

    res = {}
    base_tr = F.postprocess(tr, Ps_tr, 0.7)
    base_va = F.postprocess(va, Ps_va, 0.7)
    res['sensor'] = {'cv': score(tr, base_tr, cap_tr)}
    print(f'sensor only  CV {res["sensor"]["cv"]:.4f}')
    error_report(tr, base_tr, None, cap_tr, 'train-OOF sensor')

    cands = {}
    for path in args.images:
        name = Path(path).stem
        Ci_tr, Ci_va, Ci_te = (load_img(path, f.sample_id) for f in (tr, va, te))
        error_report(tr, base_tr, Ci_tr, cap_tr, f'train-OOF {name}')
        img_cv = score(tr, F.postprocess(tr, probs_of(Ci_tr), 0.7), cap_tr)
        row = {'img_only_cv': img_cv}
        for k in (3, 5):  # 경계 영역 AUC (Train OOF, 과거 60초 평균 이미지 P(≥k))
            X = build(tr, Ps_tr, Ci_tr, k)
            r = X['region'][cap_tr]; yy = (y_tr[cap_tr] >= k)[r]
            row[f'auc_b{k}'] = float(roc_auc_score(yy, X['i3m'][cap_tr][r])) if 0 < yy.mean() < 1 else float('nan')
        # 중첩 CV
        oof = {v: np.zeros_like(Ps_tr) for v in VARIANTS}
        rule_lab = np.zeros(len(tr), int)
        for a, b in folds:
            fa, fb = tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True)
            Xa = {k: build(fa, Ps_tr[a], Ci_tr[a], k) for k in (3, 5)}
            Xb = {k: build(fb, Ps_tr[b], Ci_tr[b], k) for k in (3, 5)}
            ms = {k: fit_gate(Xa[k], y_tr[a]) for k in (3, 5)}
            for v, ks in VARIANTS.items():
                oof[v][b] = gate_probs([ms[k] for k in ks], [Xb[k] for k in ks])
            capa = F.cap_rows(fa).index.to_numpy()
            best = max(RULE_GRID, key=lambda ct: score(fa, rule_labels(fa, Ps_tr[a], Xa[3], *ct), capa))
            rule_lab[b] = rule_labels(fb, Ps_tr[b], Xb[3], *best)
        row['rule_cv'] = score(tr, rule_lab, cap_tr)
        for v in VARIANTS:
            row[f'{v}_cv'] = score(tr, F.postprocess(tr, oof[v], 0.7), cap_tr)
        print(f'{name:28s} img {img_cv:.4f} AUC b3 {row["auc_b3"]:.3f} b5 {row["auc_b5"]:.3f} | rule {row["rule_cv"]:.4f} '
              + ' '.join(f'{v} {row[f"{v}_cv"]:.4f}' for v in VARIANTS))
        # 전체 Train으로 최종 게이트·규칙 적합 → Val/Test 적용 준비
        Xtr = {k: build(tr, Ps_tr, Ci_tr, k) for k in (3, 5)}
        ms = {k: fit_gate(Xtr[k], y_tr) for k in (3, 5)}
        row['gate_params'] = {k: {n: p.detach().numpy().round(4).tolist() for n, p in m.named_parameters()} for k, m in ms.items()}
        best = max(RULE_GRID, key=lambda ct: score(tr, rule_labels(tr, Ps_tr, Xtr[3], *ct), cap_tr))
        row['rule_ct'] = list(best)
        Xva = {k: build(va, Ps_va, Ci_va, k) for k in (3, 5)}
        Xte = {k: build(te, Ps_te, Ci_te, k) for k in (3, 5)}
        c = dict(rule=(lambda ct=best: rule_labels(va, Ps_va, Xva[3], *ct)),
                 test_rule=(lambda ct=best: rule_labels(te, Ps_te, Xte[3], *ct)), Ci_va=Ci_va)
        for v, ks in VARIANTS.items():
            c[v] = (lambda ks=ks, ms=ms, X=Xva: F.postprocess(va, gate_probs([ms[k] for k in ks], [X[k] for k in ks]), 0.7))
            c[f'test_{v}'] = (lambda ks=ks, ms=ms, X=Xte: F.postprocess(te, gate_probs([ms[k] for k in ks], [X[k] for k in ks]), 0.7))
        cands[name] = c
        res[name] = row

    # CV 1위 선택
    flat = [(res[n][f'{v}_cv'], n, v) for n in cands for v in ('rule',) + tuple(VARIANTS)]
    flat.sort(reverse=True)
    best_cv, bn, bv = flat[0]
    print(f'\nCV 1위: {bn} / {bv}  CV {best_cv:.4f}  (센서 {res["sensor"]["cv"]:.4f})')
    res['selected'] = {'image': bn, 'variant': bv, 'cv': best_cv}

    # Validation: 센서 기준선 + 선택 설정만
    res['sensor']['val'] = score(va, base_va, cap_va)
    error_report(va, base_va, cands[bn]['Ci_va'], cap_va, 'VAL sensor')
    lab = cands[bn][bv]()
    res['selected']['val'] = score(va, lab, cap_va)
    res['selected']['val_boot_diff'] = boot_diff(va, base_va, lab, cap_va)
    res['selected']['val_per_class'] = f1_score(va.label.to_numpy(int)[cap_va], lab[cap_va], labels=LAB, average=None).round(4).tolist()
    res['sensor']['val_per_class'] = f1_score(va.label.to_numpy(int)[cap_va], base_va[cap_va], labels=LAB, average=None).round(4).tolist()
    d = res['selected']['val_boot_diff']
    print(f'VAL sensor {res["sensor"]["val"]:.4f} → selected {res["selected"]["val"]:.4f}  '
          f'Δ bootstrap mean {d[0]:+.4f} 95%CI [{d[1]:+.4f}, {d[2]:+.4f}]  P(Δ>0)={d[3]:.2f}')
    if args.show_all_val:
        res['all_val_reference'] = {}
        for n in cands:
            for v in ('rule',) + tuple(VARIANTS):
                s = score(va, cands[n][v](), cap_va)
                res['all_val_reference'][f'{n}/{v}'] = s
                print(f'  [참고 Val] {n}/{v}: {s:.4f}')
    if args.write_test:
        tl = cands[bn][f'test_{bv}']()
        pd.DataFrame({'sample_id': te.sample_id, 'label': tl.astype(int)}).to_csv(args.write_test, index=False)
        print(f'wrote {args.write_test}')
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
