"""피처 영향력 분석 (모델: 다중분류 LightGBM balanced, v3 피처 735개).

1) gain 중요도 (fold 모델 평균)
2) 묶음 단위 OOF 순열 중요도: 검증 fold에서 묶음 전체를 섞었을 때 Macro F1(확신 cummax) 하락폭
   - 묶음: 신호별 / 피처 종류별 / 윈도우 길이별
3) SHAP(LightGBM pred_contrib): 실제 L2·L3 행에서 L2·L3 예측에 기여하는 피처 (2↔3 경계 원인)
4) 재학습 제거 실험 (Train CV 5-fold × 시드 3): (a) 전체 (b) 시작 대비 변화량 제외 (c) 절대 압력 제외

평가 규칙: 모두 Train CV(500행 캡)만 사용. Val은 보지 않음.
"""
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import exp_post as E
import ts_model as M

D = '/home/ubuntu/data/battery_data_light'
TAU = 0.7
RNG = np.random.default_rng(0)


def score(f, P):
    return E.f1(f.label, E._per_exp(f, P, lambda P, t: E.conf_cummax(P, TAU)))


def feature_groups(cols):
    sig_names = sorted(M.SIGNALS, key=len, reverse=True)

    def signal_of(c):
        if c in ('t1_rise_since45', 't1_rate_since45'):
            return 'meas_t1'
        if c.startswith('v_'):
            return 'meas_v'
        for s in sig_names:
            if c == s or c.startswith(s + '_'):
                return s
        return c   # relative_time_s, META

    def kind_of(c):
        s = signal_of(c)
        rest = c[len(s) + 1:] if c.startswith(s + '_') else ''
        if c in M.META or c == 'relative_time_s':
            return 'meta/time'
        if re.match(r'since\d+$', rest) or c.startswith(('t1_r', 'v_')):
            return 'crossing(임계 도달)'
        if rest == '':
            return 'raw(현재값)'
        m = re.match(r'([a-z_]+?)(\d+)$', rest)
        return f'win_{m.group(1)}' if m else rest

    def win_of(c):
        m = re.search(r'(?:mean|std|max|min|dev|slope|delta)(\d+)$', c)
        return f'window {m.group(1)}s' if m else None

    groups = {}
    for c in cols:
        groups.setdefault(('신호', signal_of(c)), []).append(c)
        groups.setdefault(('종류', kind_of(c)), []).append(c)
        w = win_of(c)
        if w:
            groups.setdefault(('윈도우', w), []).append(c)
    return groups


def main():
    pd.set_option('display.width', 250)
    tr = E.load(D, 'train')
    X_all = M.build_features(tr).reset_index(drop=True)
    trc = M.cap_rows(tr)
    X_cap = M.build_features(trc).reset_index(drop=True)
    gid = tr.experiment_id.to_numpy()
    cols = list(X_all.columns)
    groups = feature_groups(cols)

    # ---------------- 1~3: 시드 0 의 5-fold
    gain = np.zeros(len(cols)); perm = {k: [] for k in groups}; base_scores = []
    shap_abs = {2: np.zeros(len(cols)), 3: np.zeros(len(cols))}; n_shap = 0
    t0 = time.time()
    for a, b in GroupKFold(5, shuffle=True, random_state=0).split(tr, groups=gid):
        m = M.build_model('lgbm', 'balanced').fit(X_all.iloc[a], tr.label.iloc[a])
        gain += m.booster_.feature_importance('gain')
        mask = trc.experiment_id.isin(set(gid[b])).to_numpy()
        f, Xv = trc[mask], X_cap[mask].copy()
        base = score(f, m.predict_proba(Xv)); base_scores.append(base)
        for k, gc in groups.items():
            drops = []
            for _ in range(2):
                Xp = Xv.copy()
                Xp[gc] = Xv[gc].values[RNG.permutation(len(Xv))]   # 묶음 전체를 같은 순서로 섞어 묶음 내부 관계는 유지
                drops.append(base - score(f, m.predict_proba(Xp)))
            perm[k].append(np.mean(drops))
        # SHAP: 실제 L2/L3 행에서 클래스 2, 3 기여도
        sel = f.label.isin([2, 3]).to_numpy()
        contrib = m.predict_proba(Xv[sel], pred_contrib=True).reshape(sel.sum(), 6, len(cols) + 1)
        for k in (2, 3):
            shap_abs[k] += np.abs(contrib[:, k - 1, :-1]).sum(0)
        n_shap += sel.sum()
        print(f'fold done {time.time() - t0:.0f}s base {base:.4f}', flush=True)

    print(f'\n기준 CV Macro F1(시드0, 확신cummax): {np.mean(base_scores):.4f}')
    g = pd.Series(gain / gain.sum() * 100, index=cols).sort_values(ascending=False)
    print('\n=== [1] gain 중요도 상위 25 (%) ==='); print(g.head(25).round(2).to_string())
    print(f'gain 0인 피처 수: {(g == 0).sum()} / {len(g)},  누적 90%에 필요한 피처 수: {(g.cumsum() < 90).sum() + 1}')

    pt = pd.DataFrame([dict(축=k[0], 묶음=k[1], 피처수=len(groups[k]), F1하락=np.mean(v) * 100, fold표준편차=np.std(v) * 100)
                       for k, v in perm.items()])
    print('\n=== [2] 묶음 순열 중요도: 섞었을 때 Macro F1 하락 (%p, 클수록 중요 / 0 이하 = 없어도 됨) ===')
    for axis in ['신호', '종류', '윈도우']:
        print(f'\n-- {axis}별'); print(pt[pt.축 == axis].sort_values('F1하락', ascending=False).drop(columns='축').round(3).to_string(index=False))

    print('\n=== [3] SHAP: 실제 L2/L3 행에서 L2·L3 예측에 기여가 큰 피처 상위 15 (평균 |SHAP|) ===')
    s = pd.DataFrame({f'L{k}': shap_abs[k] / n_shap for k in (2, 3)}, index=cols)
    s['합'] = s.sum(1); print(s.sort_values('합', ascending=False).head(15).round(3).to_string())

    pt.to_csv('importance_groups.csv', index=False); g.to_csv('importance_gain.csv'); s.to_csv('importance_shap_L2L3.csv')

    # ---------------- 4: 재학습 제거 실험
    abs_p = ['meas_p'] + [c for c in cols if c.startswith('meas_p_') and not c.endswith('from_start')]
    ablations = {
        '(a) 전체 735': cols,
        '(b) 시작 대비 변화량(_from_start) 제외': [c for c in cols if not c.endswith('_from_start')],
        '(c) 절대 압력 제외(압력은 변화량만)': [c for c in cols if c not in abs_p],
    }
    print('\n=== [4] 재학습 제거 실험 (Train CV 5-fold × 시드 3, 확신cummax) ===', flush=True)
    for name, cs in ablations.items():
        per_seed = []
        for seed in [0, 1, 2]:
            fs, ps = [], []
            for a, b in GroupKFold(5, shuffle=True, random_state=seed).split(tr, groups=gid):
                m = M.build_model('lgbm', 'balanced').fit(X_all.iloc[a][cs], tr.label.iloc[a])
                mask = trc.experiment_id.isin(set(gid[b])).to_numpy()
                f = trc[mask]; fs.append(f)
                ps.append(E._per_exp(f, m.predict_proba(X_cap[mask][cs]), lambda P, t: E.conf_cummax(P, TAU)))
            per_seed.append(E.f1(pd.concat(fs).label, np.concatenate(ps)))
        print(f'{name:40s} 피처 {len(cs):3d} | CV {np.mean(per_seed):.4f} (시드 {" / ".join(f"{x:.4f}" for x in per_seed)})', flush=True)


if __name__ == '__main__':
    main()
