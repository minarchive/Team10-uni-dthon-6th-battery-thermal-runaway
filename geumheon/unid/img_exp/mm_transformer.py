"""Multi-modal Transformer-CNN Early Fusion 분류기 (센서 18종 + 열화상 4차원 특징 → Transformer Encoder → 6단계).

참고 구조: "Transformer Encoder(센서) + 사전학습 ResNet50 특징 추출기(이미지, 4차원 압축) Early Fusion,
          센서 Min-Max 정규화, 노이즈 제거·결측 보완". 이를 이 대회 데이터·규칙에 맞게 구현했다.

입력 토큰 (시점마다 1개, 같은 실험의 과거 --hist 행 + 현재 = 시퀀스)
  센서 18종 : meas_v, meas_t1, meas_t2, meas_p, 가스 10종, atmp_t, thermal_t, indu_t, indu_t.1
              (meas_a는 전부 0, meas_t1_plus_t2는 t1+t2 중복이라 제외)
              전처리(실험별, 인과적): 앞 값 채우기(ffill) → 뒤쪽 3행 이동 중앙값(노이즈 제거)
              → Min-Max 정규화 (fold 학습 실험의 min/max), 남은 결측은 0(=학습 최솟값)
  결측 표시 : indu_t, indu_t.1 (결측 40%/79%)
  이미지 4차원 : 고정 CNN(embed_cnn.py) 특징 → 표준화 → PCA 4 → Min-Max. 모두 fold 학습 실험으로만 fit
  조건 메타 : bat_class one-hot 3, tr_method one-hot 2, bat_cc·soc_status Min-Max
  시간      : log(1+경과초) 정규화
  → [센서 | 결측 | 이미지 | 메타 | 시간]을 이어 붙여 Linear로 d_model에 투영 (Early Fusion)
  위치 대신 '현재로부터 몇 초 전'(Δt)을 인코딩 → Test의 10/15초 간격 행에도 같은 의미

모델: Transformer Encoder (pre-norm, causal mask, 패딩 mask) → 현재(마지막) 토큰 → LayerNorm → Linear(6)
손실: 클래스 가중치(√balanced) CE + label smoothing
평가: Train 78실험 GroupKFold(5, seed 0, 다른 실험과 같은 분할). 평가 fold 실험은 500행 캡을 먼저 적용하고
      캡된 행만으로 전처리·시퀀스를 만든다(Test 조건). 후처리 확신 cummax τ=0.7. Val/Test는 fold 모델 평균.

    python mm_transformer.py --data_dir <data> --img resnet50 --out mmt_r50
    python mm_transformer.py --data_dir <data> --img none     --out mmt_noimg
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402

SENSORS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p'] + F.GAS + ['atmp_t', 'thermal_t', 'indu_t', 'indu_t.1']
MASK_COLS = ['indu_t', 'indu_t.1']
EMB = {'resnet50': 'emb_resnet50.npz', 'effb0': 'emb_effb0.npz'}
T_NORM = np.log1p(7200.0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 데이터 표: 원본 Train + (긴 실험의) 캡 사본 + Val(캡) + Test
# ---------------------------------------------------------------------------
def build_table(data_dir):
    rd = lambda s: pd.read_csv(data_dir / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    tr, va, te = rd('train'), rd('validation'), rd('test')
    parts = [tr.assign(grp=tr.experiment_id, kind='orig', orig_sid=tr.sample_id)]
    for eid, e in tr.groupby('experiment_id', sort=False):
        if len(e) > F.MAX_ROWS:
            c = F.cap_rows(e)
            parts.append(c.assign(grp=eid, kind='cap', orig_sid=c.sample_id,
                                  experiment_id=eid + '__cap', sample_id=c.sample_id + '__cap'))
    va = F.cap_rows(va)
    parts.append(va.assign(grp=va.experiment_id, kind='val', orig_sid=va.sample_id))
    parts.append(te.assign(grp=te.experiment_id, kind='test', orig_sid=te.sample_id, label=-1))
    df = pd.concat(parts, ignore_index=True)
    df['relative_time_s'] = pd.to_numeric(df.relative_time_s)
    return df


def exp_groups(df):
    t = df.relative_time_s.to_numpy(float)
    eid = df.experiment_id.to_numpy()
    return t, [idx[np.argsort(t[idx], kind='mergesort')] for idx in pd.Series(eid).groupby(eid, sort=False).indices.values()]


def clean_sensors(df, smooth):
    """실험별 인과적 전처리: ffill → 뒤쪽 smooth행 이동 중앙값. 앞부분 결측은 NaN으로 남김."""
    X = df[SENSORS].apply(pd.to_numeric, errors='coerce').to_numpy(float).copy()
    miss = np.isnan(X[:, [SENSORS.index(c) for c in MASK_COLS]]).astype(np.float32)
    _, groups = exp_groups(df)
    for idx in groups:
        x = pd.DataFrame(X[idx]).ffill()
        if smooth > 1:
            x = x.rolling(smooth, min_periods=1).median()
        X[idx] = x.to_numpy()
    return X, miss


def build_hist(df, L):
    t, groups = exp_groups(df)
    H = np.full((len(df), L), -1, np.int64)
    for idx in groups:
        for r in range(len(idx)):
            seg = idx[max(0, r - L + 1):r + 1]
            H[idx[r], L - len(seg):] = seg
    return H


def eval_rows(df, grps):
    has_cap = set(df.grp[df.kind == 'cap'])
    m = df.grp.isin(grps) & (((df.kind == 'orig') & ~df.grp.isin(has_cap)) | (df.kind == 'cap'))
    return np.where(m)[0]


# ---------------------------------------------------------------------------
# fold 전처리 (학습 실험 원본 행으로만 fit)
# ---------------------------------------------------------------------------
class FoldPrep:
    def __init__(self, Xs, meta, emb, fit_rows, img_dim):
        self.lo, self.hi = np.nanmin(Xs[fit_rows], 0), np.nanmax(Xs[fit_rows], 0)
        self.lo, self.hi = np.nan_to_num(self.lo), np.nan_to_num(self.hi, nan=1.0)
        self.m_lo, self.m_hi = meta[fit_rows].min(0), meta[fit_rows].max(0)
        self.img = None
        if emb is not None and img_dim > 0:
            E = torch.from_numpy(emb[fit_rows]).to(DEV)
            self.e_mu, self.e_sd = E.mean(0, keepdim=True), E.std(0, keepdim=True) + 1e-6
            Z = (E - self.e_mu) / self.e_sd
            _, _, V = torch.pca_lowrank(Z, q=img_dim, center=True, niter=4)
            self.V, self.z_mu = V[:, :img_dim], Z.mean(0, keepdim=True)
            P = (Z - self.z_mu) @ self.V
            self.p_lo, self.p_hi = P.min(0).values, P.max(0).values
            self.img = True

    def __call__(self, Xs, miss, meta, emb, tnorm):
        S = (Xs - self.lo) / np.maximum(self.hi - self.lo, 1e-6)
        S = np.nan_to_num(np.clip(S, -1.0, 3.0), nan=0.0)
        M = (meta - self.m_lo) / np.maximum(self.m_hi - self.m_lo, 1e-6)
        parts = [torch.from_numpy(S.astype(np.float32)), torch.from_numpy(miss), torch.from_numpy(M.astype(np.float32)),
                 torch.from_numpy(tnorm[:, None].astype(np.float32))]
        out = torch.cat(parts, 1).to(DEV)
        if self.img:
            E = torch.from_numpy(emb).to(DEV)
            P = (((E - self.e_mu) / self.e_sd) - self.z_mu) @ self.V
            P = ((P - self.p_lo) / (self.p_hi - self.p_lo).clamp(min=1e-6)).clamp(-1, 2)
            out = torch.cat([out, P], 1)
        return out.float()

    @classmethod
    def from_state(cls, st):
        """체크포인트의 prep 상태로 복원 (재학습 없이 추론)."""
        self = cls.__new__(cls)
        for k, v in st.items():
            setattr(self, k, v.to(DEV) if torch.is_tensor(v) else v)
        self.img = True if st['img'] else None
        return self

    def state(self):
        """추론 재현용 전처리 상태 (CPU 텐서/넘파이)."""
        st = dict(lo=self.lo, hi=self.hi, m_lo=self.m_lo, m_hi=self.m_hi, img=bool(self.img))
        if self.img:
            st.update({k: getattr(self, k).detach().cpu() for k in ('e_mu', 'e_sd', 'z_mu', 'V', 'p_lo', 'p_hi')})
        return st


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class MMTransformer(nn.Module):
    def __init__(self, in_dim, L, d=128, layers=4, heads=4, ff=256, drop=0.1):
        super().__init__()
        self.heads = heads
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        self.tenc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=drop, activation='gelu', batch_first=True, norm_first=True)
        self.tx = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 6))
        self.register_buffer('causal', torch.triu(torch.ones(L, L, dtype=torch.bool), 1), persistent=False)
        self.register_buffer('eye', torch.eye(L, dtype=torch.bool), persistent=False)

    def forward(self, tok, T, hist, cur):
        pad = hist < 0
        j = hist.clamp(min=0)
        dt = (T[cur][:, None] - T[j]).clamp(min=0)
        x = self.inp(tok[j]) + self.tenc(torch.stack([dt / 300, torch.log1p(dt / 5), pad.float()], -1))
        m = (self.causal[None] | pad[:, None, :]) & ~self.eye[None]
        h = self.tx(x, mask=m.repeat_interleave(self.heads, 0))
        return self.head(self.norm(h[:, -1]))


@torch.no_grad()
def predict(model, rows, tok, T, H, bs=1024):
    model.eval()
    out = []
    for j in range(0, len(rows), bs):
        cur = torch.from_numpy(rows[j:j + bs]).to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == 'cuda'):
            out.append(model(tok, T, H[cur], cur).float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def f1_post(frame, P):
    y = frame.label.astype(int).to_numpy()
    return f1_score(y, F.postprocess(frame, P, 0.7), labels=F.LABELS, average='macro')


def train_fold(a, df, X, miss, meta, emb, tnorm, T, H, tr_grps, ev, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit = np.where((df.kind == 'orig') & df.grp.isin(tr_grps))[0]
    prep = FoldPrep(X, meta, emb, fit, a.img_dim)
    tok = prep(X, miss, meta, emb, tnorm)
    y = df.label.to_numpy(int)
    cnt = np.bincount(y[fit] - 1, minlength=6).astype(float)
    cw = torch.tensor(np.sqrt(cnt.sum() / (6 * np.maximum(cnt, 1))), dtype=torch.float32, device=DEV)
    model = MMTransformer(tok.shape[1], a.hist, a.d, a.layers, a.heads, 2 * a.d, a.drop).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    groups = [fit[df.grp.to_numpy()[fit] == g] for g in np.unique(df.grp.to_numpy()[fit])]
    steps = int(np.ceil(sum(min(len(g), a.per_exp) for g in groups) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps, pct_start=0.1)
    ycu = torch.from_numpy(y - 1).to(DEV)
    ev_frame = df.iloc[ev].reset_index(drop=True)
    curve = []
    for ep in range(a.epochs):
        rows = rng.permutation(np.concatenate([rng.choice(g, min(len(g), a.per_exp), replace=False) for g in groups]))
        model.train(); tl = 0.0
        for j in range(0, steps * a.bs, a.bs):
            cur = torch.from_numpy(rows[j:j + a.bs]).to(DEV)
            if len(cur) == 0:
                break
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == 'cuda'):
                z = model(tok, T, H[cur], cur)
            loss = Fnn.cross_entropy(z.float(), ycu[cur], weight=cw, label_smoothing=a.ls)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item() * len(cur)
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            f = f1_post(ev_frame, predict(model, ev, tok, T, H))
            curve.append(round(f, 4))
            log(f'    ep{ep+1:02d} loss {tl/len(rows):.4f}  fold F1(post) {f:.4f}')
    return model, tok, curve, prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--img', default='resnet50', choices=['resnet50', 'effb0', 'none'])
    ap.add_argument('--img_dim', type=int, default=4)
    ap.add_argument('--smooth', type=int, default=3, help='뒤쪽 이동 중앙값 행 수 (1 = 노이즈 제거 안 함)')
    ap.add_argument('--hist', type=int, default=64)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--drop', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--ls', type=float, default=0.05)
    ap.add_argument('--per_exp', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--max_folds', type=int, default=5, help='빠른 점검용')
    ap.add_argument('--folds', default=None, help='이 fold만 학습 (예: 2 또는 0,3). fold 병렬 실행용')
    ap.add_argument('--merge_only', action='store_true', help='저장된 fold 결과만 합쳐서 채점')
    a = ap.parse_args()
    log(vars(a))
    t0 = time.time()
    df = build_table(a.data_dir)
    X, miss = clean_sensors(df, a.smooth)
    meta = np.stack([(df.bat_class.astype(int) == k).astype(float) for k in (1, 2, 3)] +
                    [(df.tr_method.astype(int) == k).astype(float) for k in (1, 2)] +
                    [df.bat_cc.astype(float), df.soc_status.astype(float)], 1)
    tnorm = np.log1p(df.relative_time_s.to_numpy(float).clip(min=0)) / T_NORM
    emb = None
    if a.img != 'none':
        z = np.load(EMB[a.img])
        pos = pd.Series(np.arange(len(z['sample_id'])), index=z['sample_id'].astype(str))
        emb = z['feat'][pos.loc[df.orig_sid.astype(str).values].to_numpy()].astype(np.float32)
    H = torch.from_numpy(build_hist(df, a.hist)).to(DEV)
    T = torch.tensor(df.relative_time_s.to_numpy(np.float32)).to(DEV)
    log(f'table {len(df)} rows {df.kind.value_counts().to_dict()}, sensors {len(SENSORS)}, '
        f'image {"none" if emb is None else f"{emb.shape[1]}→{a.img_dim}"}  ({time.time()-t0:.0f}s)')

    orig = df[df.kind == 'orig']
    tr_grps = orig.grp.unique()
    splits = [(set(orig.grp.iloc[ia]), set(orig.grp.iloc[ib]))
              for ia, ib in GroupKFold(5, shuffle=True, random_state=0).split(orig, groups=orig.grp.to_numpy())]
    va_rows, te_rows = np.where(df.kind == 'val')[0], np.where(df.kind == 'test')[0]
    parts = Path(f'{a.out}_parts'); parts.mkdir(exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',')]
    nf = min(a.max_folds, 5)
    run_folds = range(nf) if a.folds is None else [int(f) for f in a.folds.split(',')]
    if not a.merge_only:
        for seed in seeds:
            for k in run_folds:
                ga, gb = splits[k]
                ev = eval_rows(df, gb)
                log(f'=== seed {seed} fold {k} (train {len(ga)} exps, eval {len(gb)} exps / {len(ev)} rows) ===')
                model, tok, curve, prep = train_fold(a, df, X, miss, meta, emb, tnorm, T, H, ga, ev, seed * 100 + k)
                log(f'  params {sum(p.numel() for p in model.parameters())/1e6:.2f}M, token dim {tok.shape[1]}, '
                    f'peak GPU {(torch.cuda.max_memory_allocated() if DEV == 'cuda' else 0)/2**30:.2f} GiB')
                np.savez(parts / f's{seed}_f{k}.npz', ev=ev, P=predict(model, ev, tok, T, H),
                         Pv=predict(model, va_rows, tok, T, H), Pt=predict(model, te_rows, tok, T, H), curve=np.array(curve))
                ck = Path(f'{a.out}_models'); ck.mkdir(exist_ok=True)
                torch.save({'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()},
                            'prep': prep.state(), 'args': vars(a) | {'data_dir': str(a.data_dir), 'out': str(a.out)},
                            'model_kw': dict(in_dim=int(tok.shape[1]), L=a.hist, d=a.d, layers=a.layers, heads=a.heads,
                                             ff=2 * a.d, drop=a.drop),
                            'sensors': SENSORS, 'mask_cols': MASK_COLS,
                            'img_extractor': {'resnet50': 'timm resnet50.a1_in1k', 'effb0': 'timm efficientnet_b0.ra_in1k',
                                              'none': None}[a.img],
                            'fold': k, 'seed': seed, 'fold_train_exps': sorted(ga), 'curve': curve},
                           ck / f's{seed}_f{k}.pt')
                log(f'  saved fold part s{seed}_f{k} + checkpoint {ck}/s{seed}_f{k}.pt ({time.time()-t0:.0f}s)')
                del model, tok, prep
                if DEV == 'cuda':
                    torch.cuda.empty_cache()
    res = {'args': {k: str(v) for k, v in vars(a).items()}, 'runs': []}
    save = {}
    for seed in seeds:
        fs = [parts / f's{seed}_f{k}.npz' for k in range(nf)]
        if not all(f.exists() for f in fs):
            log(f'seed {seed}: {sum(f.exists() for f in fs)}/{nf} fold parts ready, merge later')
            continue
        Z = [np.load(f) for f in fs]
        rows = np.concatenate([z['ev'] for z in Z])
        P = np.concatenate([z['P'] for z in Z])
        Pv, Pt = np.mean([z['Pv'] for z in Z], 0), np.mean([z['Pt'] for z in Z], 0)
        curves = [z['curve'].tolist() for z in Z]
        folds = [c[-1] for c in curves]
        fr = df.iloc[rows].reset_index(drop=True)
        vf = df.iloc[va_rows].reset_index(drop=True)
        y, yv = fr.label.astype(int).to_numpy(), vf.label.astype(int).to_numpy()
        lab, labv = F.postprocess(fr, P, 0.7), F.postprocess(vf, Pv, 0.7)
        run = dict(seed=seed, cv_post=float(f1_score(y, lab, labels=F.LABELS, average='macro')),
                   cv_raw=float(f1_score(y, F.LABELS[P.argmax(1)], labels=F.LABELS, average='macro')),
                   cv_per_class=f1_score(y, lab, labels=F.LABELS, average=None).round(3).tolist(),
                   val_post=float(f1_score(yv, labv, labels=F.LABELS, average='macro')),
                   val_per_class=f1_score(yv, labv, labels=F.LABELS, average=None).round(3).tolist(),
                   fold=folds, curves=curves)
        res['runs'].append(run)
        log(f'seed {seed}: CV post {run["cv_post"]:.4f} (raw {run["cv_raw"]:.4f}) folds {folds} per-class {run["cv_per_class"]} | '
            f'Val post {run["val_post"]:.4f} {run["val_per_class"]}')
        save.update({f's{seed}_oof_sid': fr.orig_sid.to_numpy(str), f's{seed}_oof': P, f's{seed}_val': Pv, f's{seed}_test': Pt})
    if not res['runs']:
        return
    cv = [r['cv_post'] for r in res['runs']]; vv = [r['val_post'] for r in res['runs']]
    res.update(cv_mean=float(np.mean(cv)), cv_std=float(np.std(cv)), val_mean=float(np.mean(vv)), val_std=float(np.std(vv)))
    log(f'CV post {res["cv_mean"]:.4f} ± {res["cv_std"]:.4f} | Val post {res["val_mean"]:.4f} ± {res["val_std"]:.4f} '
        f'over {len(cv)} seeds  ({time.time()-t0:.0f}s)')
    Path(f'res_{a.out.name}.json').write_text(json.dumps(res, indent=1))
    np.savez(a.out.with_name(a.out.name + '.npz'), val_sid=df.orig_sid.to_numpy(str)[va_rows],
             test_sid=df.orig_sid.to_numpy(str)[te_rows], **save)

if __name__ == '__main__':
    main()
