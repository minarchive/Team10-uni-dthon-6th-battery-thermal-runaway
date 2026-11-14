"""순서형(ordinal) 분해 vs 6클래스 다중분류.

학습: 각 분류기는 원래 피처 + 정답 라벨로 직접 학습 (다른 모델 출력을 입력으로 쓰지 않음 → OOF 불필요)
평가 규칙
- 선택: Train 78개 실험 GroupKFold(5, shuffle) × 시드 3개, 검증 fold 500행 캡
- Val : CV로 고른 후보와 기준 모델만 Macro F1로 확인 (Val로 고르지 않음)

ordinal: k=2..6 에 대해 이진 분류기 'y >= k' 5개.
  누적확률 c_k 를 k 방향으로 단조 감소하게 보정(cummin) → P(y=k) = c_k - c_{k+1}
  디코딩 argmax: 클래스 확률 argmax / thr: c_k > 0.5 인 최대 k
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import exp_post as E
import report as R
import ts_model as M

D = '/home/ubuntu/data/battery_data_light'
SEEDS = [0, 1, 2]
TAU = 0.7


def lgbm(cw):
    return M.build_model('lgbm', cw)


class Ordinal:
    def __init__(self, cw='balanced'):
        self.cw = cw

    def fit(self, X, y):
        y = np.asarray(y)
        self.models = [lgbm(self.cw).fit(X, (y >= k).astype(int)) for k in range(2, 7)]
        return self

    def predict_proba(self, X):
        C = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])   # c2..c6
        C = np.minimum.accumulate(C, axis=1)                                    # c2 >= c3 >= ... >= c6
        full = np.column_stack([np.ones(len(C)), C, np.zeros(len(C))])          # c1=1, c7=0
        return full[:, :-1] - full[:, 1:]


def decode(frame, P, how, post):
    if how == 'argmax':
        fn = (lambda P, t: E.conf_cummax(P, TAU)) if post else (lambda P, t: E.L[P.argmax(1)])
        return E._per_exp(frame, P, fn)
    # thr: c_k = P(y>=k) > 0.5 인 최대 k
    C = np.cumsum(P[:, ::-1], axis=1)[:, ::-1]
    lab = 1 + (C[:, 1:] > 0.5).sum(1)
    if not post:
        return lab
    # 확신도 = 그 단계 '이상'일 확률(1단계는 2단계 미만일 확률). conf_cummax 입력 형식으로 변환
    conf = np.where(lab == 1, 1 - C[:, 1], C[np.arange(len(C)), lab - 1])
    Q = np.zeros_like(P); Q[np.arange(len(P)), lab - 1] = conf
    return E._per_exp(frame, Q, lambda P, t: E.conf_cummax(P, TAU))


VARIANTS = {   # 이름: (모델 생성, 디코딩 목록)
    'multiclass': (lambda: lgbm('balanced'), ['argmax']),
    'ordinal-balanced': (lambda: Ordinal('balanced'), ['argmax', 'thr']),
    'ordinal-none': (lambda: Ordinal(None), ['argmax', 'thr']),
}


def main():
    tr, va = E.load(D, 'train'), E.load(D, 'validation')
    X_all = M.build_features(tr).reset_index(drop=True)          # 실험별 독립 계산 → 실험 단위로 잘라 써도 동일
    trc = M.cap_rows(tr)
    X_cap = M.build_features(trc).reset_index(drop=True)
    gid = tr.experiment_id.to_numpy()

    oof = {}   # (variant, decode, post) -> list of (seed, frame, pred)
    for vname, (make, decs) in VARIANTS.items():
        t0 = time.time()
        for seed in SEEDS:
            for a, b in GroupKFold(5, shuffle=True, random_state=seed).split(tr, groups=gid):
                m = make().fit(X_all.iloc[a], tr.label.iloc[a])
                mask = trc.experiment_id.isin(set(gid[b])).to_numpy()
                f, P = trc[mask], m.predict_proba(X_cap[mask])
                for dec in decs:
                    for post in [False, True]:
                        oof.setdefault((vname, dec, post), []).append((seed, f, decode(f, P, dec, post)))
        print(f'{vname}: {time.time() - t0:.0f}s', flush=True)

    rows = []
    for key, items in oof.items():
        per_seed = []
        for s in SEEDS:
            f = pd.concat([x[1] for x in items if x[0] == s]); p = np.concatenate([x[2] for x in items if x[0] == s])
            per_seed.append(E.f1(f.label, p))
        rows.append(dict(model=key[0], decode=key[1], post='확신cummax' if key[2] else 'raw',
                         cv=np.mean(per_seed), cv_seed_std=np.std(per_seed),
                         seeds=' / '.join(f'{x:.4f}' for x in per_seed)))
    res = pd.DataFrame(rows).sort_values('cv', ascending=False)
    pd.set_option('display.width', 250)
    print('\n=== Train CV Macro F1 (GroupKFold 5 × 시드 3, 500행 캡) ===')
    print(res.round(4).to_string(index=False), flush=True)

    # CV 1위 후보 + 기준(multiclass 확신cummax)만 Val 확인
    best = res.iloc[0]
    cands = [('multiclass', 'argmax', True)]
    if (best.model, best.decode, best.post == '확신cummax') not in cands:
        cands.append((best.model, best.decode, best.post == '확신cummax'))
    vc = M.cap_rows(va); Xv = M.build_features(vc)[X_all.columns]
    for vname, dec, post in cands:
        tag = f'{vname} / {dec} / {"확신cummax" if post else "raw"}'
        items = [x for x in oof[(vname, dec, post)] if x[0] == 0]
        R.print_tables(pd.concat([x[1] for x in items]), np.concatenate([x[2] for x in items]), f'CV(시드0 OOF) — {tag}')
        m = VARIANTS[vname][0]().fit(X_all, tr.label)
        p = decode(vc, m.predict_proba(Xv), dec, post)
        print(f'\n[Val 확인] {tag}: Macro F1 {E.f1(vc.label, p):.4f}')
        R.print_tables(vc, p, f'Val — {tag}')


if __name__ == '__main__':
    main()
