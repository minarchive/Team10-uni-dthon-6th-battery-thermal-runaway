"""Causal sequence experiments: duration-aware filtering and frozen-DINO GRU.

No PCA is used. Splits are by experiment; all temporal features use current/past
rows only. The single seed-0 GroupKFold run is a screening pass.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'val0.9195'))
import features as F  # noqa: E402
from exp_img import img_features, lgbm, load, log  # noqa: E402

LABELS = np.arange(1, 7)
N_SPLITS = 5
SEED = 0
GRU_EPOCHS = 8
GRU_BATCH = 4
GRU_HIDDEN = 128
EMB_PROJ = 64
DEVICE = 'cuda'


def progress(path, phase, pct, detail):
    data = {'phase': phase, 'pct': float(pct), 'detail': detail, 'updated': time.time()}
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def make_features(frame, emb):
    ts = F.build_features(frame)
    cos = img_features(frame, emb, None, 32, use_pca=False, use_cos=True)
    return pd.concat([ts, cos], axis=1).loc[frame.sample_id.astype(str).values]


def sequences(frame, X, E):
    d = frame.reset_index(drop=True)
    t = d.relative_time_s.to_numpy(float)
    out = []
    for exp, ix in d.groupby('experiment_id', sort=False).indices.items():
        ix = ix[np.argsort(t[ix], kind='mergesort')]
        ids = d.sample_id.iloc[ix].astype(str).values
        out.append({
            'experiment_id': str(exp),
            'rows': ix,
            'x': X.loc[ids].to_numpy(np.float32),
            'e': E.loc[ids].to_numpy(np.float32),
            'y': d.label.iloc[ix].to_numpy(np.int64) - 1 if 'label' in d else None,
        })
    return out


def fit_standardizer(seqs):
    x = np.concatenate([s['x'] for s in seqs])
    med = np.nanmedian(x, axis=0)
    med[~np.isfinite(med)] = 0
    x = np.where(np.isfinite(x), x, med)
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-6] = 1
    return med.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def apply_standardizer(seqs, stats):
    med, mean, scale = stats
    out = []
    for s in seqs:
        x = np.where(np.isfinite(s['x']), s['x'], med)
        out.append({**s, 'x': ((x - mean) / scale).astype(np.float32)})
    return out


def collate(items):
    lengths = torch.tensor([len(s['x']) for s in items], dtype=torch.long)
    n, t, dx, de = len(items), int(lengths.max()), items[0]['x'].shape[1], items[0]['e'].shape[1]
    x = torch.zeros(n, t, dx, dtype=torch.float32)
    e = torch.zeros(n, t, de, dtype=torch.float32)
    y = torch.full((n, t), -100, dtype=torch.long)
    for i, s in enumerate(items):
        m = len(s['x'])
        x[i, :m] = torch.from_numpy(s['x'])
        e[i, :m] = torch.from_numpy(s['e'])
        if s['y'] is not None:
            y[i, :m] = torch.from_numpy(s['y'])
    return x, e, y, lengths


class CausalDinoGRU(nn.Module):
    def __init__(self, x_dim, emb_dim):
        super().__init__()
        self.img_norm = nn.LayerNorm(emb_dim)
        self.img_proj = nn.Sequential(nn.Linear(emb_dim, EMB_PROJ), nn.GELU(), nn.Dropout(0.15))
        self.gru = nn.GRU(x_dim + EMB_PROJ, GRU_HIDDEN, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(GRU_HIDDEN, 6)

    def forward(self, x, e):
        image = self.img_proj(self.img_norm(e))
        h, _ = self.gru(torch.cat([x, image], dim=-1))
        return self.head(self.dropout(h))


def fit_gru(train_seqs, valid_seqs, stats, fold, model_dir, epochs=GRU_EPOCHS):
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    random.seed(SEED + fold)
    torch.cuda.manual_seed_all(SEED + fold)
    tr = apply_standardizer(train_seqs, stats)
    model = CausalDinoGRU(tr[0]['x'].shape[1], tr[0]['e'].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.03)
    y = np.concatenate([s['y'] for s in tr])
    counts = np.bincount(y, minlength=6).astype(np.float32)
    cw = np.sqrt(len(y) / (6 * np.maximum(counts, 1)))
    cw /= cw.mean()
    cw = torch.tensor(cw, device=DEVICE)
    order_rng = np.random.default_rng(SEED + fold)
    for ep in range(epochs):
        model.train()
        order = order_rng.permutation(len(tr))
        losses = []
        for start in range(0, len(order), GRU_BATCH):
            items = [tr[i] for i in order[start:start + GRU_BATCH]]
            x, e, yy, lengths = collate(items)
            x, e, yy = x.to(DEVICE), e.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x, e).float()
            flat = nn.functional.cross_entropy(logits.flatten(0, 1), yy.flatten(), weight=cw,
                                               ignore_index=-100, reduction='none').view_as(yy)
            mask = yy.ne(-100)
            per_seq = (flat * mask).sum(1) / mask.sum(1).clamp_min(1)
            loss = per_seq.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach()))
        log(f'fold {fold} ep {ep + 1}/{epochs} loss {np.mean(losses):.4f}')
    model_dir.mkdir(parents=True, exist_ok=True)
    stats_np = {'median': stats[0], 'mean': stats[1], 'scale': stats[2]}
    ckpt = model_dir / f'gru_frozen_fold{fold}.pt'
    tmp = ckpt.with_name(ckpt.name + '.tmp')
    torch.save({'model_type': 'causal_dino_gru_frozen', 'state_dict': model.state_dict(),
                'x_dim': tr[0]['x'].shape[1], 'emb_dim': tr[0]['e'].shape[1],
                'hidden': GRU_HIDDEN, 'projection': EMB_PROJ, 'fold': fold,
                'stats': stats_np, 'epochs': epochs, 'seed': SEED + fold}, tmp)
    tmp.replace(ckpt)
    return model, apply_standardizer(valid_seqs, stats), stats


@torch.no_grad()
def predict_gru(model, seqs):
    model.eval()
    predictions = []
    for start in range(0, len(seqs), GRU_BATCH):
        items = seqs[start:start + GRU_BATCH]
        x, e, _, lengths = collate(items)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(x.to(DEVICE), e.to(DEVICE)).float().softmax(-1).cpu().numpy()
        predictions.extend(p[i, :int(lengths[i])] for i in range(len(items)))
    return predictions


def sequence_probs(frame, seqs, ps):
    out = np.full((len(frame), 6), np.nan, np.float32)
    for s, p in zip(seqs, ps):
        out[s['rows']] = p
    return out


def metrics(frame, probs, filtered=False):
    cap = F.cap_rows(frame.reset_index(drop=True))
    ix = cap.index.to_numpy()
    cap = cap.reset_index(drop=True)
    P = probs[ix]
    y = cap.label.astype(int).to_numpy()
    pred = LABELS[P.argmax(1)]
    out = {'raw': float(f1_score(y, pred, labels=LABELS, average='macro'))}
    out['cummax'] = float(f1_score(y, F.postprocess(cap, P, 0.7), labels=LABELS, average='macro'))
    if filtered:
        out['per_class'] = f1_score(y, pred, labels=LABELS, average=None).round(4).tolist()
    return out


def duration_model(frame, step_s):
    """Estimate monotone adjacent-stage duration hazards; last run is right-censored."""
    d = frame.reset_index(drop=True)
    max_dur = 0.0
    samples = [[] for _ in range(6)]
    bad = 0
    for _, ix in d.groupby('experiment_id', sort=False).indices.items():
        ix = ix[np.argsort(d.relative_time_s.to_numpy(float)[ix], kind='mergesort')]
        t, y = d.relative_time_s.to_numpy(float)[ix], d.label.to_numpy(int)[ix]
        starts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1]
        ends = np.r_[starts[1:], len(y)]
        for j, (a, b) in enumerate(zip(starts, ends)):
            state = int(y[a]) - 1
            if state < 0 or state > 5:
                continue
            if j + 1 < len(starts):
                nxt = int(y[starts[j + 1]])
                if nxt < int(y[a]):
                    bad += 1
                    continue
                dur, event = float(t[starts[j + 1]] - t[a]), True
            else:
                dur, event = float(t[b - 1] - t[a] + step_s), False
            dur = max(dur, step_s)
            samples[state].append((dur, event))
            max_dur = max(max_dur, dur)
    n_age = int(np.clip(np.ceil(max_dur / step_s) + 2, 32, 4096))
    hazards = np.zeros((6, n_age), np.float32)
    for s, obs in enumerate(samples[:5]):
        if not obs:
            hazards[s, :] = 1 / max(2, n_age)
            continue
        bins = [(min(n_age - 1, max(0, int(np.floor(dur / step_s)))), ev) for dur, ev in obs]
        previous = 0.01
        for age in range(n_age):
            risk = sum(b >= age for b, _ in bins)
            events = sum(b == age and ev for b, ev in bins)
            if risk:
                previous = float((events + 0.5) / (risk + 1.0))
            hazards[s, age] = np.clip(previous, 1e-4, 0.5)
    hazards[5] = 0.0
    first = np.bincount(d.groupby('experiment_id', sort=False).head(1).label.astype(int) - 1,
                        minlength=6).astype(np.float64) + 0.5
    first /= first.sum()
    return {'hazards': hazards, 'first': first.astype(np.float32), 'step_s': float(step_s),
            'n_age': n_age, 'nonmonotone_runs': bad}


def filter_one(P, times, model):
    h, step = model['hazards'], model['step_s']
    n_age = h.shape[1]
    alpha = np.zeros((6, n_age), np.float64)
    alpha[:, 0] = model['first']
    alpha[:, 0] *= np.maximum(P[0], 1e-8)
    alpha /= max(alpha.sum(), 1e-12)
    out = np.zeros_like(P, dtype=np.float32)
    out[0] = alpha.sum(1)
    for i in range(1, len(P)):
        steps = max(0, int(round((times[i] - times[i - 1]) / step)))
        for _ in range(steps):
            nxt = np.zeros_like(alpha)
            for s in range(6):
                if n_age > 1:
                    nxt[s, 1:] += alpha[s, :-1] * (1 - h[s, :-1])
                nxt[s, -1] += alpha[s, -1] * (1 - h[s, -1])
                if s < 5:
                    nxt[s + 1, 0] += np.dot(alpha[s], h[s])
                else:
                    nxt[s, -1] += np.dot(alpha[s, :-1], h[s, :-1])
            alpha = nxt
            z = alpha.sum()
            if z > 0:
                alpha /= z
        alpha *= np.maximum(P[i], 1e-8)[:, None]
        z = alpha.sum()
        alpha /= max(z, 1e-12)
        out[i] = alpha.sum(1)
    return out


def filter_frame(frame, probs, model):
    d = frame.reset_index(drop=True)
    out = np.full_like(probs, np.nan, dtype=np.float32)
    t = d.relative_time_s.to_numpy(float)
    for _, ix in d.groupby('experiment_id', sort=False).indices.items():
        ix = ix[np.argsort(t[ix], kind='mergesort')]
        out[ix] = filter_one(probs[ix], t[ix], model)
    return out


def score_filtered(frame, probs):
    cap = F.cap_rows(frame.reset_index(drop=True))
    ix = cap.index.to_numpy()
    cap = cap.reset_index(drop=True)
    P = probs[ix]
    y = cap.label.astype(int).to_numpy()
    pred = LABELS[P.argmax(1)]
    return {'filtered': float(f1_score(y, pred, labels=LABELS, average='macro')),
            'per_class': f1_score(y, pred, labels=LABELS, average=None).round(4).tolist()}


def save_json(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--emb', type=Path, default=Path('emb_dinov2s.npz'))
    ap.add_argument('--out_dir', type=Path, default=Path('.'))
    ap.add_argument('--epochs', type=int, default=GRU_EPOCHS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prog = args.out_dir / 'seq_progress.json'
    tr, va, te = [load(args.data_dir, s).reset_index(drop=True) for s in ['train', 'validation', 'test']]
    z = np.load(args.emb)
    ids = z['sample_id'].astype(str)
    E = pd.DataFrame(np.concatenate([z['cls'], z['patch_mean']], axis=1).astype(np.float32), index=ids)
    Xtr, Xva, Xte = [make_features(f, E) for f in (tr, va, te)]
    log(f'features ready: train={Xtr.shape}, val={Xva.shape}, test={Xte.shape}; no PCA')

    # Fold-safe baseline OOF and online duration filtering.
    progress(prog, 'HSMM base OOF', 0, 'v3+cos OOF 확률 및 Train 체류 시간 추정')
    oof_base = np.full((len(tr), 6), np.nan, np.float32)
    oof_hsmm = np.full_like(oof_base, np.nan)
    fold_hazards = []
    splits = list(GroupKFold(N_SPLITS, shuffle=True, random_state=SEED).split(tr, groups=tr.experiment_id))
    for fold, (a, b) in enumerate(splits):
        model = lgbm(SEED)
        model.fit(Xtr.iloc[a], tr.label.iloc[a].astype(int))
        p = model.predict_proba(Xtr.iloc[b]).astype(np.float32)
        oof_base[b] = p
        ev = tr.iloc[b].reset_index(drop=True)
        ts = np.median(np.concatenate([np.diff(g.relative_time_s.to_numpy(float)) for _, g in tr.iloc[a].groupby('experiment_id') if len(g) > 1]))
        dm = duration_model(tr.iloc[a].reset_index(drop=True), step_s=float(np.clip(ts, 1, 10)))
        oof_hsmm[b] = filter_frame(ev, p, dm)
        fold_hazards.append(dm)
        log(f'HSMM/base fold {fold + 1}/{N_SPLITS}: nonmonotone train runs={dm["nonmonotone_runs"]}')
        progress(prog, 'HSMM base OOF', 20 * (fold + 1), f'fold {fold + 1}/{N_SPLITS}')

    full_model = lgbm(42).fit(Xtr, tr.label.astype(int))
    val_base = full_model.predict_proba(Xva).astype(np.float32)
    test_base = full_model.predict_proba(Xte).astype(np.float32)
    dts = np.concatenate([np.diff(g.relative_time_s.to_numpy(float)) for _, g in tr.groupby('experiment_id') if len(g) > 1])
    dm_full = duration_model(tr, float(np.clip(np.median(dts), 1, 10)))
    val_hsmm = filter_frame(va, val_base, dm_full)
    test_hsmm = filter_frame(te, test_base, dm_full)
    np.savez(args.out_dir / 'seq_base_oof.npz', sample_id=tr.sample_id.to_numpy(str), probs=oof_base, hsmm=oof_hsmm)
    np.savez(args.out_dir / 'seq_base_valtest.npz', val_sample_id=va.sample_id.to_numpy(str), val_probs=val_base,
             val_hsmm=val_hsmm, test_sample_id=te.sample_id.to_numpy(str), test_probs=test_base, test_hsmm=test_hsmm)
    base_scores = {'cv/v3+cos': metrics(tr, oof_base), 'cv/hsmm': score_filtered(tr, oof_hsmm),
                   'val/v3+cos': metrics(va, val_base), 'val/hsmm': score_filtered(va, val_hsmm)}
    save_json(args.out_dir / 'res_hsmm.json', base_scores)
    log(f'HSMM done: {base_scores}')

    # Frozen DINOv2-S embeddings with a learned 64-d projection and causal GRU.
    progress(prog, 'Frozen DINO-S GRU', 20, '실험 단위 GroupKFold(5), frozen encoder, PCA 없음')
    tr_seqs = sequences(tr, Xtr, E)
    va_seqs = sequences(va, Xva, E)
    te_seqs = sequences(te, Xte, E)
    oof_gru = np.full((len(tr), 6), np.nan, np.float32)
    val_gru = np.zeros((len(va), 6), np.float32)
    test_gru = np.zeros((len(te), 6), np.float32)
    model_dir = args.out_dir / 'seq_models'
    for fold, (a, b) in enumerate(splits):
        tr_ids, ev_ids = set(tr.experiment_id.iloc[a]), set(tr.experiment_id.iloc[b])
        fold_tr = [s for s in tr_seqs if s['experiment_id'] in tr_ids]
        fold_ev = [s for s in tr_seqs if s['experiment_id'] in ev_ids]
        stats = fit_standardizer(fold_tr)
        model, ev_std, _ = fit_gru(fold_tr, fold_ev, stats, fold, model_dir, args.epochs)
        p_ev = predict_gru(model, ev_std)
        # Sequence rows are local to the fold frame; map by sample ID to full Train order.
        p_by_id = {sid: p for s, ps in zip(ev_std, p_ev)
                   for sid, p in zip(tr.sample_id.iloc[s['rows']].astype(str), ps)}
        oof_gru[b] = np.stack([p_by_id[sid] for sid in tr.sample_id.iloc[b].astype(str)])
        for target_seqs, target_frame, store in [(va_seqs, va, 'val'), (te_seqs, te, 'test')]:
            target_std = apply_standardizer(target_seqs, stats)
            ps = predict_gru(model, target_std)
            pp = sequence_probs(target_frame, target_std, ps)
            if store == 'val': val_gru += pp / N_SPLITS
            else: test_gru += pp / N_SPLITS
        del model
        torch.cuda.empty_cache()
        log(f'GRU fold {fold + 1}/{N_SPLITS} complete')
        progress(prog, 'Frozen DINO-S GRU', 20 + 16 * (fold + 1), f'fold {fold + 1}/{N_SPLITS}, {args.epochs} epochs')

    cv_blend = (oof_base + oof_gru) / 2
    cv_hybrid = (oof_hsmm + oof_gru) / 2
    val_blend = (val_base + val_gru) / 2
    val_hybrid = (val_hsmm + val_gru) / 2
    res = {
        **base_scores,
        'cv/gru_frozen': metrics(tr, oof_gru),
        'cv/avg_v3cos_gru': metrics(tr, cv_blend),
        'cv/avg_hsmm_gru': metrics(tr, cv_hybrid),
        'val/gru_frozen': metrics(va, val_gru),
        'val/avg_v3cos_gru': metrics(va, val_blend),
        'val/avg_hsmm_gru': metrics(va, val_hybrid),
    }
    save_json(args.out_dir / 'res_sequence.json', res)
    np.savez(args.out_dir / 'seq_predictions.npz', train_sample_id=tr.sample_id.to_numpy(str), base_oof=oof_base,
             hsmm_oof=oof_hsmm, gru_oof=oof_gru, val_sample_id=va.sample_id.to_numpy(str), val_base=val_base,
             val_hsmm=val_hsmm, val_gru=val_gru, test_sample_id=te.sample_id.to_numpy(str), test_base=test_base,
             test_hsmm=test_hsmm, test_gru=test_gru)
    log(f'Sequence experiments complete: {res}')
    progress(prog, '완료', 100, 'HSMM 및 frozen DINO-S GRU 완료. 결과 검토 후 부분 encoder 미세조정 여부 결정')


if __name__ == '__main__':
    main()
