"""두 계열 센서 모델 + 학습된 이미지 백본 앙상블 (CV 최대화, 과충전·2→3 따로 관리).

평가 틀 (모든 구성 모델 공통, 해당 실험을 학습에 쓰지 않은 예측만 사용)
  CV  : Train 78개 실험 GroupKFold(5, seed 0) OOF, 500행 캡
  Val : Train 전체 학습 모델의 Val 19개 실험 예측, 500행 캡
  → 97개 실험을 모은 OOS 확률에서 앙상블 구성을 고른다.

구성 후보
  센서(우리 계열, 837 피처·45°C 규칙·과충전 증강): xgb_R_J6wide, lgb_L63, xgb_J12wide, xgb_J6wide, xgb_v3A, xgb_v3A_cB(DINOv2-B 코사인 피처)
  센서(SOTA 계열, 166 피처·과충전 C 분기)       : lgb, et, lgb_p23(2→3 피처), lgb_bal
  이미지(학습된 백본)                            : eff(EfficientNet-B0 차이 영상), cnx(ConvNeXt-T), dino_s12·dino_b4(DINOv2 파인튜닝), tstack

방법
  1) 일반 실험(tr_method 1)과 과충전(tr_method 2)의 가중치를 따로 고른다 — 탐욕적 앙상블 선택(중복 허용, 디코딩 후 macro F1 최대화)
  2) 규칙: t1 누적최대 < 45°C → 1단계, ≥ 45°C → 1단계 금지, < 50°C → 3단계 이상 금지 (Train·Val 97개 실험 100% 성립)
  3) 디코더: 인과 단조 Viterbi(SOTA) 와 확신 cummax(τ 0.7) 중 nested 점수가 높은 쪽
  4) 정직한 점수: 97개 실험 5-fold nested — 가중치는 다른 fold 실험에서 고르고 빠진 fold 에 적용. CV 부분/Val 부분을 나눠 보고
  5) 최종 가중치는 97개 전체로 선택 → Test 확률(Train+Val 학습 모델) 결합 → blend_sota/ 에 설정·제출 파일·가중치 저장
"""
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')
from final_oc import F, load, D  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'blend_sota'; OUT.mkdir(exist_ok=True)
REPO = Path('/home/nxcolelxu/Team10-uni-dthon-6th-battery-thermal-runaway')
LAB = np.arange(1, 7)
MAX_PICK = int(sys.argv[1]) if len(sys.argv) > 1 else 10
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ 데이터·은행
tr, va, te = [load(D, s).reset_index(drop=True) for s in ('train', 'validation', 'test')]
for f in (tr, va, te):
    f['sample_id'] = f.sample_id.astype(str)
tr_cap, va_cap = F.cap_rows(tr).reset_index(drop=True), F.cap_rows(va).reset_index(drop=True)
fr = pd.concat([tr_cap.assign(_src='cv'), va_cap.assign(_src='val')], ignore_index=True)     # 97개 실험 OOS 프레임
n_cv = len(tr_cap)
BANK = {}


def add(name, cv, val, test, weights=None):
    assert cv.shape == (len(tr_cap), 6) and val.shape == (len(va_cap), 6) and test.shape == (len(te), 6), name
    BANK[name] = dict(oos=np.r_[cv, val].astype(np.float64), test=test.astype(np.float64), weights=weights)


def npz_ids(path, ids_key='sample_id', p_key='probs'):
    z = np.load(path, allow_pickle=True)
    return pd.DataFrame(z[p_key], index=z[ids_key].astype(str))


def try_add(fn):
    try:
        fn()
    except Exception as e:  # 아직 안 만들어진 구성은 건너뜀
        log(f'  skip: {type(e).__name__} {str(e)[:90]}')


def ours(name, d, cfg, wglob):
    add(name, np.load(HERE / d / f'oof_{cfg}.npy'), np.load(HERE / d / f'valprobs_{cfg}.npy'), np.load(HERE / d / f'test_probs_{cfg}.npy'),
        weights=sorted((HERE / d / 'ckpt' / cfg).glob(wglob)))


try_add(lambda: ours('xgb_R_J6wide', 'final_oc4_gpu', 'R_J6wide', 'final_seed*.joblib'))
try_add(lambda: ours('lgb_L63', 'final_oc6_oc6_lgb63', 'L_J6wide_L63', 'final_seed*.joblib'))
try_add(lambda: ours('xgb_J12wide', 'final_oc3_oc3c', 'J12wide', 'final_seed*.joblib'))
try_add(lambda: ours('xgb_J6wide', 'final_oc3_oc3a', 'J6wide', 'final_seed*.joblib'))
for _n, _d, _c in (('xgb_J2ref', 'final_oc3_oc3a', 'J2ref'), ('xgb_J6thin', 'final_oc3_oc3a', 'J6thin'), ('xgb_J12xwide', 'final_oc3_oc3c', 'J12xwide'),
                   ('xgb_R_J2ref', 'final_oc4_gpu', 'R_J2ref'), ('xgb_R_noreset', 'final_oc4_gpu', 'R_J6wide_noreset'),
                   ('xgb_R_B63', 'final_oc5_oc5_base', 'R_J6wide_B63'), ('xgb_R_ood', 'final_oc5_oc5_ood', 'R_J6wide')):
    try_add(lambda _n=_n, _d=_d, _c=_c: ours(_n, _d, _c, 'final_seed*.joblib'))
for nm, cfg in (('xgb_v3A', 'v3A'), ('xgb_v3A_cB', 'v3A_cB'), ('xgb_v3A_cBS', 'v3A_cBS'), ('xgb_v3A_cB_capaug', 'v3A_cB_capaug')):
    def _mm(nm=nm, cfg=cfg):
        z = np.load(HERE / 'final_mm_gpu' / f'oof_{cfg.replace("_", "+")}.npz')
        add(nm, z['oof'], z['val'], np.load(HERE / 'final_mm_gpu' / f'test_probs_{cfg}.npy'), weights=sorted((HERE / 'final_mm_gpu' / f'models_{cfg}').glob('*')))
    try_add(_mm)


def _sota():
    z = np.load(HERE / 'sota_bank.npz', allow_pickle=True)
    ti, vi, xi = z['tr_ids'].astype(str), z['va_ids'].astype(str), z['te_ids'].astype(str)
    for m in ('lgb', 'et', 'lgb_p23', 'lgb_bal'):
        if f'{m}_test' not in z.files:
            continue
        g = lambda ids, P, tgt: pd.DataFrame(P, index=ids).loc[tgt].to_numpy()
        add(m, g(ti, z[f'{m}_cv'], tr_cap.sample_id.values), g(vi, z[f'{m}_val'], va_cap.sample_id.values), g(xi, z[f'{m}_test'], te.sample_id.values),
            weights=[HERE.parent / 'blend_sota' / 'weights' / f'{m}_final_trval.joblib'])


try_add(_sota)


def img(name, oos_npz, test_npz, weights):
    a = npz_ids(HERE / oos_npz); b = npz_ids(HERE / test_npz)
    add(name, a.loc[tr_cap.sample_id.values].to_numpy(), a.loc[va_cap.sample_id.values].to_numpy(), b.loc[te.sample_id.values].to_numpy(), weights=weights)


try_add(lambda: img('eff', 'eff_diff_b0.npz', 'eff_diff_b0_trval.npz', sorted((REPO / 'v3_sota_et0.8920' / 'weights').glob('vision_seed*.pt'))))
try_add(lambda: img('cnx', 'ft_convnext_tiny.npz', 'ft_convnext_tiny_trval.npz', sorted((HERE / 'ft_convnext_tiny_trval_models').glob('*.pt'))))
try_add(lambda: img('dino_s12', 'ft_ord_dinov2s_u12.npz', 'ft_ord_dinov2s_u12.npz', sorted((HERE / 'ft_ord_dinov2s_u12_models').glob('*.pt'))))
try_add(lambda: img('dino_b4', 'ft_ord_dinov2b_u4.npz', 'ft_ord_dinov2b_u4.npz', sorted((HERE / 'ft_ord_dinov2b_u4_models').glob('*.pt'))))
try_add(lambda: add('tstack', np.load(HERE / 'temporal_stack/oof_tr_cap.npy'), np.load(HERE / 'temporal_stack/val_va_cap.npy'), np.load(HERE / 'temporal_stack/test.npy'),
                    weights=[HERE / 'temporal_stack/ckpt/final_trval.pt']))
NAMES = list(BANK)
log(f'bank: {NAMES}')


# ------------------------------------------------------------------ 규칙·디코더
def exp_index(frame):
    t = frame.relative_time_s.to_numpy(float); out = {}
    for e, idx in frame.groupby('experiment_id', sort=False).indices.items():
        out[e] = idx[np.argsort(t[idx], kind='mergesort')]
    return out


def t1cmax(frame, EI):
    x = pd.to_numeric(frame.meas_t1, errors='coerce').to_numpy(float); out = np.empty(len(frame))
    for idx in EI.values():
        out[idx] = np.fmax.accumulate(np.nan_to_num(x[idx], nan=-np.inf))
    return out


def rules(P, c):
    P = P.copy()
    lo = c < 45
    P[lo] = 0; P[lo, 0] = 1
    P[~lo, 0] = 0
    P[c < 50, 2:] = 0
    s = P.sum(1)
    bad = s <= 0
    if bad.any():
        P[bad & (c < 50), 1] = 1; P[bad & (c >= 50), 1:] = 1 / 5
        s = P.sum(1)
    return P / s[:, None]


def decode(P, EI, exps, kind):
    out = np.zeros(len(P), int)
    for e in exps:
        idx = EI[e]; p = P[idx]
        if kind == 'viterbi':
            L = np.log(np.clip(p, 1e-6, 1)); best = L[0]; lab = np.empty(len(idx), int); lab[0] = best.argmax() + 1
            for j in range(1, len(idx)):
                best = np.maximum.accumulate(best) + L[j]; lab[j] = best.argmax() + 1
            out[idx] = lab
        else:
            out[idx] = F.conf_cummax(p, 0.7)
    return out


EI = exp_index(fr); C = t1cmax(fr, EI)
EI_te = exp_index(te); C_te = t1cmax(te, EI_te)


def corr23(P, q, c):
    """2/3 확률 배분을 2↔3 전용 판별 모델의 q = P(≥3) 로 교체. 3~6단계 사이의 비율은 원래 모델 그대로."""
    out = P.copy(); h = c >= 45; s_ = P[:, 2:].sum(1)
    out[~h] = 0; out[~h, 0] = 1
    out[h, 0] = 0; out[h, 1] = 1 - q[h]
    sc = np.where(s_ > 1e-9, q / np.maximum(s_, 1e-9), 0.0)
    out[h, 2:] = P[h, 2:] * sc[h, None]
    z = h & (s_ <= 1e-9); out[z, 2] = q[z]
    return out


def _spec():
    z = np.load(HERE / 'p23_spec.npz', allow_pickle=True)
    g = lambda ids, v, tgt: pd.Series(v, index=ids.astype(str)).loc[tgt].to_numpy()
    kinds = [k for k in ('lgb', 'et') if f'{k}_test' in z.files]
    q_oos = np.mean([np.r_[g(z['tr_ids'], z[f'{k}_cv'], tr_cap.sample_id.values), g(z['va_ids'], z[f'{k}_val'], va_cap.sample_id.values)] for k in kinds], 0)
    q_te = np.mean([g(z['te_ids'], z[f'{k}_test'], te.sample_id.values) for k in kinds], 0)
    wf = sorted((HERE.parent / 'blend_sota' / 'weights').glob('p23spec_*_final_trval.joblib'))
    for base in ('lgb', 'et', 'xgb_J12wide', 'xgb_v3A', 'lgb_L63'):
        if base in BANK:
            BANK[f'{base}+s23'] = dict(oos=corr23(BANK[base]['oos'], q_oos, C), test=corr23(BANK[base]['test'], q_te, C_te), weights=list(BANK[base]['weights'] or []) + wf)
    log(f'  2↔3 판별 모델 {kinds} 보정 구성 추가')


try_add(_spec)
NAMES = list(BANK)
RAW = {k: dict(oos=v['oos'], test=v['test']) for k, v in BANK.items()}
T_FR = fr.relative_time_s.to_numpy(float); T_TE = te.relative_time_s.to_numpy(float)


def smooth(P, EIx, t, Wsec):
    """같은 실험의 최근 Wsec 초(현재 포함, 과거만) 확률 평균"""
    if Wsec <= 0:
        return P
    out = np.empty_like(P)
    for idx in EIx.values():
        tt = t[idx]; cs = np.cumsum(P[idx], 0); j = np.searchsorted(tt, tt - Wsec, side='right'); i = np.arange(len(idx))
        prev = np.where(j[:, None] > 0, cs[np.maximum(j - 1, 0)], 0.0)
        out[idx] = (cs - prev) / (i - j + 1)[:, None]
    return out


def set_smooth(Wsec):
    for k in NAMES:
        BANK[k]['oos'] = smooth(RAW[k]['oos'], EI, T_FR, Wsec); BANK[k]['test'] = smooth(RAW[k]['test'], EI_te, T_TE, Wsec)
y = fr.label.astype(int).to_numpy()
EXP = pd.DataFrame({'tm': fr.groupby('experiment_id', sort=False).tr_method.first(), 'src': fr.groupby('experiment_id', sort=False)._src.first()})
_n = pd.concat([tr, va]).groupby('experiment_id').size()
EXP['long'] = _n.reindex(EXP.index).to_numpy() > 500          # Test 의 긴 실험처럼 500행으로 다운샘플되는 실험
ROWS = lambda exps: np.concatenate([EI[e] for e in exps])


def f1(lab, rows, labels=None):
    yy = y[rows]
    return f1_score(yy, lab[rows], labels=labels if labels is not None else np.unique(yy), average='macro')


def obj(lab, exps):
    """선택 기준. 과충전 그룹은 Test 구성(과충전 행의 77%가 500행으로 다운샘플된 긴 실험)에 맞춰 긴 과충전 F1 을 절반 반영."""
    sc = f1(lab, ROWS(exps))
    if EXP.tm[exps[0]] == 2:
        longs = [e for e in exps if EXP.long[e]]
        if longs:
            sc = 0.5 * sc + 0.5 * f1(lab, ROWS(longs))
    return sc


def blend(counts, key='oos'):
    tot = sum(counts.values())
    return sum(BANK[k][key] * (v / tot) for k, v in counts.items())


def greedy(exps, kind, max_pick=MAX_PICK):
    """exps(같은 그룹) 위에서 디코딩 후 macro F1 을 최대화하는 구성 (중복 허용 전진 선택)."""
    rows = ROWS(exps); S = np.zeros((len(fr), 6)); counts = {}; best_hist = []
    for step in range(max_pick):
        cand = []
        for k in NAMES:
            P = rules((S + BANK[k]['oos']) / (step + 1), C)
            cand.append((obj(decode(P, EI, exps, kind), exps), k))
        sc, k = max(cand)
        S += BANK[k]['oos']; counts[k] = counts.get(k, 0) + 1
        best_hist.append((sc, dict(counts)))
    sc, cnt = max(best_hist, key=lambda x: (round(x[0], 5), -sum(x[1].values())))
    return cnt, sc


def lag3(lab, exps):
    out = []
    t = fr.relative_time_s.to_numpy(float)
    for e in exps:
        idx = EI[e]; a = np.where(lab[idx] >= 3)[0]; b = np.where(y[idx] >= 3)[0]
        if len(a) and len(b):
            out.append(t[idx[a[0]]] - t[idx[b[0]]])
    return np.median(out), np.mean(out)


def report(name, lab):
    cv_e = EXP.index[EXP.src == 'cv']; va_e = EXP.index[EXP.src == 'val']; oc_e = EXP.index[EXP.tm == 2]
    r = dict(cv=f1(lab, ROWS(cv_e), LAB), cv_oc=f1(lab, ROWS(cv_e.intersection(oc_e)), LAB), val=f1(lab, ROWS(va_e), LAB),
             val_oc=f1(lab, ROWS(va_e.intersection(oc_e)), LAB), all97=f1(lab, np.arange(len(fr)), LAB), oc97=f1(lab, ROWS(oc_e), LAB))
    ocl_e = EXP.index[(EXP.tm == 2) & EXP.long]; r['oc_long'] = f1(lab, ROWS(ocl_e))
    pc = f1_score(y, lab, labels=LAB, average=None)
    l_cv, l_va = lag3(lab, cv_e), lag3(lab, va_e)
    log(f'{name:34s} CV {r["cv"]:.4f} (OC {r["cv_oc"]:.4f}) | Val {r["val"]:.4f} (OC {r["val_oc"]:.4f}) | 97개 {r["all97"]:.4f} (OC {r["oc97"]:.4f}) '
        f'| 긴 과충전 {r["oc_long"]:.4f} | F1 2단계 {pc[1]:.3f} 3단계 {pc[2]:.3f} | 2→3 지연 CV {l_cv[0]:+.0f}/{l_cv[1]:+.0f} Val {l_va[0]:+.0f}/{l_va[1]:+.0f}')
    r.update(f1_2=float(pc[1]), f1_3=float(pc[2]), per_class=pc.round(4).tolist())
    return r


# ------------------------------------------------------------------ 단일 모델·기준
RES = {}
allx = list(EXP.index)
log('== 단일 구성 (규칙 적용, Viterbi)')
for k in NAMES:
    RES[f'single/{k}'] = report(k, decode(rules(BANK[k]['oos'], C), EI, allx, 'viterbi'))
if {'xgb_R_J6wide', 'cnx'} <= set(NAMES):
    P = 0.9 * BANK['xgb_R_J6wide']['oos'] + 0.1 * BANK['cnx']['oos']
    RES['ref/oc4_R_J6wide+cnx0.1'] = report('기준 oc4 R_J6wide+cnx0.1 (cummax)', decode(P, EI, allx, 'cummax'))
if {'lgb', 'et', 'eff'} <= set(NAMES):
    P = 0.94 * (BANK['lgb']['oos'] + BANK['et']['oos']) / 2 + 0.06 * BANK['eff']['oos']
    RES['ref/sota_et'] = report('기준 SOTA_ET 구조 (lgb+et, eff 0.06)', decode(P, EI, allx, 'viterbi'))
    RES['ref/sota_et+rules'] = report('  〃 + 규칙', decode(rules(P, C), EI, allx, 'viterbi'))

# ------------------------------------------------------------------ 선택 방법 2: 고정 레시피 (자유도 작음)
OURS = [k for k in ('xgb_R_J6wide', 'lgb_L63', 'xgb_J12wide', 'xgb_J6wide', 'xgb_v3A', 'xgb_v3A_cB') if k in BANK]
SOTA2 = [k for k in ('lgb', 'et') if k in BANK]
SOTA4 = [k for k in ('lgb', 'et', 'lgb_p23', 'lgb_bal') if k in BANK]
IMGS = {n: [k for k in ks if k in BANK] for n, ks in (('eff', ['eff']), ('cnx', ['cnx']), ('eff+cnx', ['eff', 'cnx']), ('eff+cnx+dino', ['eff', 'cnx', 'dino_s12']))}
RECIPES = {}
SOTA2S = [k for k in ('lgb', 'et', 'lgb+s23', 'et+s23') if k in BANK]
for sn, sota in (('sota2', SOTA2), ('sota4', SOTA4), ('sota2s', SOTA2S)):
    for ws in (0.0, 0.3, 0.5, 0.7, 1.0):                       # SOTA 계열 비중
        if (ws > 0 and not sota) or (ws < 1 and not OURS) or (ws == 0 and sn != 'sota2'):
            continue
        sens = {k: (1 - ws) / len(OURS) for k in OURS} if ws < 1 else {}
        if ws > 0:
            sens.update({k: ws / len(sota) for k in sota})
        RECIPES[f'{sn}:{ws}|img:none'] = sens
        for inm, iks in IMGS.items():
            if not iks:
                continue
            for al in (0.06, 0.1, 0.15):
                w = {k: v * (1 - al) for k, v in sens.items()}; w.update({k: al / len(iks) for k in iks})
                RECIPES[f'{sn}:{ws}|img:{inm}@{al}'] = w
log(f'recipes {len(RECIPES)}')
REC_LAB = {}


def build_labs(kinds):
    REC_LAB.clear()
    RP = {n: rules(blend(w), C) for n, w in RECIPES.items()}
    for kind_ in kinds:
        REC_LAB[kind_] = {n: decode(P, EI, allx, kind_) for n, P in RP.items()}


def pick_recipe(exps, kind, only=None):
    names = [n for n in RECIPES if only is None or only(n)]
    sc, n = max((round(obj(REC_LAB[kind][n], exps), 5), n) for n in names)
    return RECIPES[n], sc, n


def pick_greedy(exps, kind):
    cnt, sc = greedy(exps, kind)
    return cnt, sc, 'greedy'


def oc_sota(exps, kind):
    """과충전: SOTA 계열(C 분기) 센서만 + 이미지 소량. Test 과충전은 시작 압력이 학습 범위 밖이라 CV 로는 검증이 안 되므로
    Public 에서 검증된 구조로 제한하고, 그 안에서만 비중을 고른다."""
    return pick_recipe(exps, kind, only=lambda n: n.split('|')[0].endswith(':1.0'))


def by_tm(sel1, sel2):
    return lambda exps, kind: (sel2 if EXP.tm[exps[0]] == 2 else sel1)(exps, kind)


SELECTORS = {'A 제한없음(greedy)': by_tm(pick_greedy, pick_greedy), 'B tm1 greedy + 과충전 SOTA계열': by_tm(pick_greedy, oc_sota),
             'C tm1 recipe + 과충전 SOTA계열': by_tm(pick_recipe, oc_sota)}

# ------------------------------------------------------------------ nested 비교 (가중치는 다른 fold 실험에서 선택)
folds = list(GroupKFold(5, shuffle=True, random_state=0).split(allx, groups=allx))
best = None
for Wsm in (0, 15, 30):
    set_smooth(Wsm)
    kinds = ('viterbi', 'cummax') if Wsm == 0 else ('viterbi',)
    build_labs(kinds)
    for sel_name, sel in SELECTORS.items():
        if Wsm > 0 and sel_name.startswith('A'):
            continue
        for kind in kinds:
            lab = np.zeros(len(fr), int)
            for a, b in folds:
                tr_e = [allx[i] for i in a]; ev_e = [allx[i] for i in b]
                for tm in (1, 2):
                    w, _, _ = sel([e for e in tr_e if EXP.tm[e] == tm], kind)
                    ev = [e for e in ev_e if EXP.tm[e] == tm]
                    if ev:
                        lab_ = decode(rules(blend(w), C), EI, ev, kind); rows = ROWS(ev); lab[rows] = lab_[rows]
            r = report(f'nested {sel_name} ({kind}, 평활 {Wsm}s)', lab); RES[f'nested/{sel_name}/{kind}/smooth{Wsm}'] = r
            if not sel_name.startswith('A') and (best is None or r['all97'] > best[2]['all97']):   # 최종은 과충전 제한이 있는 B/C 중에서
                best = (sel_name, kind, r, sel, Wsm)
sel_name, kind, _, sel, Wsm = best
set_smooth(Wsm); build_labs((kind,))
log(f'선택 방법: {sel_name}, 디코더: {kind}, 평활 {Wsm}s')

# ------------------------------------------------------------------ 최종 선택 (97개 전체)
W = {}; PICK = {}
lab = np.zeros(len(fr), int)
for tm in (1, 2):
    ex = [e for e in allx if EXP.tm[e] == tm]
    cnt, sc, nm = sel(ex, kind); W[tm] = cnt; PICK[tm] = nm
    lab_ = decode(rules(blend(cnt), C), EI, ex, kind); rows = ROWS(ex); lab[rows] = lab_[rows]
    tot = sum(cnt.values()); log(f'최종 tr_method={tm} [{nm}]: ' + ', '.join(f'{k} {v / tot:.3f}' for k, v in sorted(cnt.items(), key=lambda x: -x[1])) + f'  (그룹 F1 {sc:.4f})')
RES['final_in_sample'] = report('최종 앙상블 (선택에 쓴 97개, 낙관적)', lab)
if True:                                                        # 상위 레시피 표 (참고)
    for tm in (1, 2):
        ex = [e for e in allx if EXP.tm[e] == tm]; rows = ROWS(ex)
        top = sorted(((obj(REC_LAB[kind][n], ex), n) for n in RECIPES), reverse=True)[:8]
        log(f'  tr_method={tm} 상위 레시피: ' + ' / '.join(f'{n} {s_:.4f}' for s_, n in top))

# ------------------------------------------------------------------ Test
Pt = np.zeros((len(te), 6)); tm2 = (te.tr_method == 2).to_numpy()
Pt[~tm2] = blend(W[1], 'test')[~tm2]; Pt[tm2] = blend(W[2], 'test')[tm2]
Pt = rules(Pt, C_te)
lab_te = decode(Pt, EI_te, list(EI_te), kind)
np.save(OUT / 'test_probs.npy', Pt)
ss = pd.read_csv(D / 'sample_submission.csv', dtype={'sample_id': str})
sub = ss[['sample_id']].merge(pd.DataFrame({'sample_id': te.sample_id.values, 'label': lab_te}), on='sample_id', how='left')
assert sub.label.notna().all() and len(sub) == len(ss)
sub.label = sub.label.astype(int); sub.to_csv(OUT / 'submission.csv', index=False)
log(f'submission 분포 {sub.label.value_counts().sort_index().to_dict()}')
ref = {k: pd.read_csv(REPO / p).set_index('sample_id').label.loc[te.sample_id.values].to_numpy() for k, p in
       (('SOTA_ET 0.8920', 'v3_sota_et0.8920/submission.csv'), ('SOTA 0.8812', 'sota_0.8812/submission.csv'),
        ('oc4 R_J6wide', 'geumheon/unid/img_exp/final_oc4_gpu/submission_R_J6wide_convnext0.1.csv'))}
agree = {k: dict(diff=int((v != lab_te).sum()), f1=float(f1_score(v, lab_te, average='macro'))) for k, v in ref.items()}
log('Test 비교: ' + ' | '.join(f'{k}: 다른 행 {v["diff"]}, 일치도 {v["f1"]:.4f}' for k, v in agree.items()))
t = te.relative_time_s.to_numpy(float); rows = []
for e, idx in EI_te.items():
    r = dict(e=e[:6], tm=int(te.tr_method.iloc[idx[0]]), n=len(idx), diff=int((lab_te[idx] != ref['SOTA_ET 0.8920'][idx]).sum()))
    for L_ in (3, 4, 5):
        on = lambda lab_: t[idx[np.argmax(lab_[idx] >= L_)]] if (lab_[idx] >= L_).any() else np.nan
        r[f'd{L_}'] = on(lab_te) - on(ref['SOTA_ET 0.8920'])
    rows.append(r)
print(pd.DataFrame(rows).sort_values('diff', ascending=False).to_string(index=False), flush=True)

# ------------------------------------------------------------------ 저장
used = sorted(set(W[1]) | set(W[2]))
wdir = OUT / 'weights'; wdir.mkdir(exist_ok=True); manifest = {}
for k in used:
    dst = wdir / k; files = []
    for p in BANK[k]['weights'] or []:
        p = Path(p)
        if not p.exists():
            continue
        if p.parent == wdir:
            files.append(p.name); continue
        dst.mkdir(exist_ok=True)
        if p.is_dir():
            shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
        else:
            shutil.copy2(p, dst / p.name)
        files.append(f'{k}/{p.name}')
    manifest[k] = files
cfg = dict(decoder=kind, smooth_sec=Wsm, rules=['t1 cummax < 45C -> stage 1', 't1 cummax >= 45C -> not stage 1', 't1 cummax < 50C -> stage <= 2'],
           weights={f'tr_method_{tm}': {k: v / sum(c.values()) for k, v in c.items()} for tm, c in W.items()},
           selector=sel_name, picked={f'tr_method_{tm}': n for tm, n in PICK.items()}, components=manifest, test_agreement=agree)
(OUT / 'config.json').write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
(OUT / 'results.json').write_text(json.dumps(RES, indent=1, ensure_ascii=False))
log(f'saved → {OUT}  (구성 {used})')
