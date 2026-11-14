"""센서 단독 인과 트랜스포머 + 원시 시계열 증강 비교 실험.

목적: joint_moe2의 센서 인코더(SensorDecoder)가 LightGBM보다 약한 원인이 실험 수(78개) 부족에 의한 과적합인지,
     원시 시계열 증강으로 줄일 수 있는지 확인한다. 이미지는 쓰지 않는다.

증강 (원시 CSV 값에 적용한 뒤 features.build_features로 피처를 다시 계산. 실험 단위로 하나의 변환)
  - time warp  : relative_time_s × s, s ~ U(1-tw, 1+tw). 행과 라벨이 함께 움직이므로 전환 시점(행)은 보존
  - row drop   : t=0 행을 뺀 나머지 행을 확률 p ~ U(0, drop)로 제거 (불규칙·성긴 샘플링 흉내)
  - gas/p/atmp : 채널별 배율 exp(N(0, gas_sd)) + 행별 상대 노이즈 (라벨 정의와 무관한 채널)
  - 온도·전압  : t2/thermal_t/indu_t 에 N(0, temp_sd)°C, meas_v 에 N(0, v_sd)V 노이즈
  - meas_t1    : 변경하지 않음 (1→2 라벨 = t1 누적최대 45°C 최초 도달, Train·Val 100% 규칙)
  - 하지 않음  : 시작 구간 자르기(모든 실험이 0초 시작, 누적 피처가 시작점 의존), 실험 간 mixup

누수 방지
  - 증강본은 원본 실험과 같은 그룹 → 원본이 학습 fold일 때만 학습에 쓰인다
  - SensorPrep(표준화·결측 열)은 fold 학습 실험의 '원본' 행으로만 fit
  - 평가 fold 실험은 500행 캡을 먼저 적용하고 그 행만으로 피처·히스토리를 계산 (Test와 같은 조건)

공정 비교: 에폭마다 실험별로 (원본 확률 p_orig, 아니면 증강본 중 하나) 버전 1개를 고르고 최대 per_exp행 추출.
         증강 유무와 상관없이 에폭당 스텝 수가 같다.

    python sensor_aug.py --data_dir <data> --out sa_noaug --n_aug 0
    python sensor_aug.py --data_dir <data> --out sa_aug   --n_aug 6
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

import joint_moe as J1
from finetune import F, log
from finetune_dino import K, cum_to_probs
from joint_moe2 import SensorDecoder, build_hist

SCALE_CH = F.GAS + ['meas_p']
NOISE_T = ['meas_t2', 'thermal_t', 'indu_t']


def augment_exp(e, rng, a):
    """실험 하나(시간순 DataFrame)에 무작위 변환 1개를 적용한 사본."""
    e = e.copy()
    if a.drop > 0 and len(e) > 20:
        p = rng.uniform(0, a.drop)
        keep = rng.random(len(e)) >= p
        keep[0] = True
        e = e[keep]
    if a.tw > 0:
        e['relative_time_s'] = e['relative_time_s'].astype(float) * rng.uniform(1 - a.tw, 1 + a.tw)
    n = len(e)
    for c in SCALE_CH:
        x = pd.to_numeric(e[c], errors='coerce')
        e[c] = x * np.exp(rng.normal(0, a.gas_sd)) * (1 + rng.normal(0, a.gas_sd / 4, n))
    if a.atmp_sd > 0:
        e['atmp_t'] = pd.to_numeric(e['atmp_t'], errors='coerce') + rng.normal(0, a.atmp_sd)
    for c in NOISE_T:
        e[c] = pd.to_numeric(e[c], errors='coerce') + rng.normal(0, a.temp_sd, n)
    e['meas_v'] = pd.to_numeric(e['meas_v'], errors='coerce') + rng.normal(0, a.v_sd, n)
    return e


def build_table(data_dir, a):
    """원본 Train + 증강 사본 + 캡된 평가용 사본 (+ Val 캡) 한 표로. 컬럼: grp(원본 실험), kind(orig/aug/cap)."""
    rd = lambda s: pd.read_csv(data_dir / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    tr = rd('train')
    rng = np.random.default_rng(a.aug_seed)
    parts = [tr.assign(grp=tr.experiment_id, kind='orig', split='train')]
    for eid, e in tr.groupby('experiment_id', sort=False):
        e = e.sort_values('relative_time_s')
        for k in range(a.n_aug):
            x = augment_exp(e, rng, a)
            parts.append(x.assign(grp=eid, kind='aug', split='train',
                                  experiment_id=f'{eid}__a{k}', sample_id=x.sample_id + f'__a{k}'))
        if len(e) > F.MAX_ROWS:
            c = F.cap_rows(e)
            parts.append(c.assign(grp=eid, kind='cap', split='train',
                                  experiment_id=f'{eid}__cap', sample_id=c.sample_id + '__cap'))
    if a.eval_val:
        va = F.cap_rows(rd('validation'))
        parts.append(va.assign(grp=va.experiment_id, kind='val', split='validation'))
    df = pd.concat(parts, ignore_index=True)
    t0 = time.time()
    X = F.build_features(df).to_numpy(np.float32)
    log(f'table {len(df)} rows ({(df.kind == "aug").sum()} augmented), features {X.shape} in {time.time()-t0:.0f}s')
    return df, X


def eval_rows(df, grp_set):
    """평가 행: 캡이 필요한 실험은 cap 사본, 아니면 원본."""
    has_cap = set(df.grp[df.kind == 'cap'])
    m = df.grp.isin(grp_set) & (((df.kind == 'orig') & ~df.grp.isin(has_cap)) | (df.kind == 'cap'))
    return np.where(m)[0]


class SensorOnly(nn.Module):
    def __init__(self, dim, L, hid=256):
        super().__init__()
        self.enc = SensorDecoder(dim, L, out=hid)
        self.head = nn.Linear(hid, K)

    def forward(self, Xp, T, H, cur):
        return self.head(self.enc(Xp, T, H[cur], cur))


@torch.no_grad()
def predict(model, rows, ctx, dev, bs=512):
    model.eval()
    out = []
    for j in range(0, len(rows), bs):
        cur = torch.from_numpy(rows[j:j + bs]).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(ctx['Xp'], ctx['T'], ctx['H'], cur).float()).cpu().numpy())
    return np.concatenate(out)


def f1_post(frame, cum):
    y = frame.label.astype(int).to_numpy()
    return f1_score(y, F.postprocess(frame, cum_to_probs(cum), 0.7), labels=F.LABELS, average='macro')


def train_fold(a, df, X, ctx_base, tr_grps, ev_rows, dev, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    orig_tr = np.where((df.kind == 'orig') & df.grp.isin(tr_grps))[0]
    prep = J1.SensorPrep(X[orig_tr])
    ctx = dict(ctx_base, Xp=torch.from_numpy(prep(X)).to(dev))
    labels = df.label.to_numpy(float).astype(int)
    pos = np.array([(labels[orig_tr] >= k).mean() for k in range(2, 7)])
    pw = torch.tensor(np.sqrt((1 - pos) / np.maximum(pos, 1e-3)), dtype=torch.float32, device=dev)
    ks = torch.arange(2, 7, device=dev).view(1, K)
    # 실험(그룹)별 버전 목록: [원본 행, 증강본1 행, ...]
    sub = df[df.grp.isin(tr_grps) & df.kind.isin(['orig', 'aug'])]
    versions = {}
    for (g, eid), idx in sub.groupby(['grp', 'experiment_id'], sort=False).indices.items():
        versions.setdefault(g, {})[eid] = sub.index.to_numpy()[idx]
    versions = {g: [v[g]] + [v[e] for e in v if e != g] for g, v in versions.items()}
    model = SensorOnly(prep.dim, a.hist).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = int(np.ceil(sum(min(len(v[0]), a.per_exp) for v in versions.values()) / a.bs)) + 5
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps, pct_start=0.1)
    ev_frame = df.iloc[ev_rows].reset_index(drop=True)
    curve, done = [], 0
    for ep in range(a.epochs):
        rows = []
        for v in versions.values():
            r = v[0] if (len(v) == 1 or rng.random() < a.p_orig) else v[rng.integers(1, len(v))]
            n = min(len(v[0]), a.per_exp)
            rows.append(rng.choice(r, min(n, len(r)), replace=False))
        rows = rng.permutation(np.concatenate(rows))
        model.train(); tl = 0.0
        for j in range(0, len(rows), a.bs):
            cur = torch.from_numpy(rows[j:j + a.bs]).to(dev)
            y = torch.from_numpy(labels[rows[j:j + a.bs]]).to(dev)
            tgt = (y.view(-1, 1) >= ks).float() * (1 - a.ls) + 0.5 * a.ls
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(ctx['Xp'], ctx['T'], ctx['H'], cur)
            loss = Fnn.binary_cross_entropy_with_logits(z.float(), tgt, pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if done < a.epochs * steps - 1:
                sched.step(); done += 1
            tl += loss.item() * len(cur)
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            f = f1_post(ev_frame, predict(model, ev_rows, ctx, dev))
            curve.append(round(f, 4))
            log(f'    ep{ep+1:02d} loss {tl/len(rows):.4f}  fold F1(post) {f:.4f}')
    return model, ctx, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--cv_seed', type=int, default=0)
    # 증강
    ap.add_argument('--n_aug', type=int, default=0, help='실험당 증강 사본 수 (0 = 증강 없음)')
    ap.add_argument('--p_orig', type=float, default=0.4, help='에폭마다 원본을 고를 확률')
    ap.add_argument('--aug_seed', type=int, default=123)
    ap.add_argument('--tw', type=float, default=0.15)
    ap.add_argument('--drop', type=float, default=0.3)
    ap.add_argument('--gas_sd', type=float, default=0.2)
    ap.add_argument('--atmp_sd', type=float, default=0.5)
    ap.add_argument('--temp_sd', type=float, default=0.3)
    ap.add_argument('--v_sd', type=float, default=0.005)
    # 모델·학습 (joint_moe2 센서 브랜치와 같은 기본값)
    ap.add_argument('--hist', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--ls', type=float, default=0.05)
    ap.add_argument('--per_exp', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=2)
    ap.add_argument('--eval_val', action='store_true', help='최종 확인용. 선택에는 쓰지 않는다')
    a = ap.parse_args()
    log(vars(a))
    dev = 'cuda'
    df, X = build_table(a.data_dir, a)
    H = build_hist(df, a.hist)
    ctx_base = dict(T=torch.tensor(df.relative_time_s.to_numpy(np.float32), device=dev), H=torch.from_numpy(H).to(dev))

    # joint_moe2 / finetune 과 같은 분할: 원본 Train 행 순서 그대로 행 단위 GroupKFold
    orig = df[df.kind == 'orig']
    tr_grps = orig.grp.unique()
    splits = []
    for ia, ib in GroupKFold(5, shuffle=True, random_state=a.cv_seed).split(orig, groups=orig.grp.to_numpy()):
        ga, gb = set(orig.grp.iloc[ia]), set(orig.grp.iloc[ib])
        splits.append((np.array([g in ga for g in tr_grps]), np.array([g in gb for g in tr_grps])))
    res = {'args': {k: str(v) for k, v in vars(a).items()}, 'runs': []}
    for seed in [int(s) for s in a.seeds.split(',')]:
        oof_rows, oof_cum, val_cum, folds, curves = [], [], [], [], []
        for k, (ia, ib) in enumerate(splits):
            ev = eval_rows(df, set(tr_grps[ib]))
            log(f'=== seed {seed} fold {k} (train {ia.sum()} exps, eval {ib.sum()} exps / {len(ev)} rows) ===')
            model, ctx, curve = train_fold(a, df, X, ctx_base, set(tr_grps[ia]), ev, dev, seed * 100 + k)
            oof_rows.append(ev); oof_cum.append(predict(model, ev, ctx, dev))
            folds.append(curve[-1]); curves.append(curve)
            if a.eval_val:
                val_cum.append(predict(model, np.where(df.kind == 'val')[0], ctx, dev))
            del model, ctx; torch.cuda.empty_cache()
        rows = np.concatenate(oof_rows)
        frame = df.iloc[rows].reset_index(drop=True)
        cum = np.concatenate(oof_cum)
        y = frame.label.astype(int).to_numpy()
        post = F.postprocess(frame, cum_to_probs(cum), 0.7)
        run = {'seed': seed, 'cv_post': f1_post(frame, cum), 'fold': folds, 'curves': curves,
               'per_class': f1_score(y, post, labels=F.LABELS, average=None).round(4).tolist()}
        if a.eval_val:
            vf = df[df.kind == 'val'].reset_index(drop=True)
            run['val_post'] = f1_post(vf, np.mean(val_cum, 0))
        log(f'seed {seed}: CV post {run["cv_post"]:.4f}  folds {folds}  per-class {run["per_class"]}'
            + (f'  Val post {run["val_post"]:.4f}' if a.eval_val else ''))
        res['runs'].append(run)
        np.savez(a.out.with_name(f'{a.out.name}_s{seed}.npz'), sample_id=frame.sample_id.to_numpy(str), cum=cum)
    cv = [r['cv_post'] for r in res['runs']]
    res['cv_mean'], res['cv_std'] = float(np.mean(cv)), float(np.std(cv))
    log(f'CV post mean {res["cv_mean"]:.4f} ± {res["cv_std"]:.4f} over {len(cv)} seeds')
    Path(f'res_{a.out.name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
