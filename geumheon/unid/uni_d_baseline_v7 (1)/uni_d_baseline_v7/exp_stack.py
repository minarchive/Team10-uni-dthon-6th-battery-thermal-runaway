"""확률 스태킹 실험: in-sample vs OOF.

1차: v3 LightGBM (model_ts_v3.joblib, Train으로 학습)
2차 피처: 같은 실험의 현재·이전 시점 1차 확률만으로 만든 피처 (초 단위, 실험별 계산)
2차 모델: LightGBM(balanced), 입력 = 원래 피처 735개 + 스태킹 피처

- in-sample: 2차 학습용 1차 확률을 '그 샘플을 학습한 1차 모델'로 만듦
- OOF      : 2차 학습용 1차 확률을 GroupKFold out-of-fold로 만듦 (probs_v3.npz)
Val/Test 추론 시 1차 확률은 두 방식 모두 Train 전체로 학습한 1차 모델로 만듭니다.
"""
import joblib
import numpy as np
import pandas as pd

import exp_post as E
import report as R
import ts_model as M

D = '/home/ubuntu/data/battery_data_light'
WIN = [30, 60, 120]


def stack_features(frame: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    """같은 실험·현재 이하 시점의 1차 확률만 사용. frame 행 순서로 반환."""
    d = pd.DataFrame(P, columns=[f'p{k}' for k in M.LABELS], index=frame.index)
    for k in range(2, 7):
        d[f'c{k}'] = d[[f'p{j}' for j in range(k, 7)]].sum(1)      # P(단계 >= k)
    d['pred'] = E.L[P.argmax(1)]
    d['experiment_id'] = frame['experiment_id'].astype(str).values
    d['t'] = pd.to_numeric(frame['relative_time_s']).values
    d = d.sort_values(['experiment_id', 't'])
    d['_ts'] = pd.to_datetime(d['t'], unit='s')
    g = d.groupby('experiment_id', sort=False)
    C = [f'c{k}' for k in range(2, 7)]
    out = d[[f'p{k}' for k in M.LABELS] + C].copy()
    for c in C:
        out[f'{c}_cummax'] = g[c].cummax()
        first = d['t'].where(d[c] >= 0.5).groupby(d['experiment_id']).cummin()
        out[f'{c}_since05'] = d['t'] - first                         # c_k가 처음 0.5 넘은 뒤 경과 시간
    for w in WIN:
        r = d.groupby('experiment_id', sort=False).rolling(f'{w}s', on='_ts', closed='right')[C]
        mean, mx = r.mean().reset_index(drop=True), r.max().reset_index(drop=True)
        for c in C:
            out[f'{c}_mean{w}'] = mean[c].values
            out[f'{c}_max{w}'] = mx[c].values
    pm = g['pred'].cummax()
    for k in range(2, 7):                                           # 예측 단계 k에 처음 도달한 뒤 경과 시간
        first = d['t'].where(pm >= k).groupby(d['experiment_id']).cummin()
        out[f'pred{k}_since'] = d['t'] - first
    return out.loc[frame.index].astype(np.float32)


def fit_stage2(X1, S, y):
    m = M.build_model('lgbm', 'balanced')
    return m.fit(pd.concat([X1.reset_index(drop=True), S.reset_index(drop=True)], axis=1), y)


def predict_stage2(m, X1, S):
    return m.predict_proba(pd.concat([X1.reset_index(drop=True), S.reset_index(drop=True)], axis=1))


def main():
    tr, va = E.load(D, 'train'), E.load(D, 'validation')
    te = pd.read_csv(f'{D}/test/test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    b = joblib.load('model_ts_v3.joblib'); m1, cols = b['model'], b['features']

    # 2차 학습 데이터 (Train, 500행 캡)
    z = np.load('probs_v3.npz')
    tr_oof = tr.loc[z['oof_idx']]; P_oof = z['oof_P']
    tr_cap = tr.loc[tr_oof.index]                       # 같은 행 집합(캡 적용된 OOF 프레임과 동일)
    X1_tr = M.build_features(tr_cap)[cols]
    P_in = m1.predict_proba(X1_tr)

    # Val / Test 1차 확률 (Train 전체 1차 모델)
    vc = M.cap_rows(va); X1_va = M.build_features(vc)[cols]; P_va = m1.predict_proba(X1_va)
    X1_te = M.build_features(te)[cols]; P_te = m1.predict_proba(X1_te)
    S_va, S_te = stack_features(vc, P_va), stack_features(te, P_te)

    base_raw = E.L[P_va.argmax(1)]
    base_post = E._per_exp(vc, P_va, lambda P, t: E.conf_cummax(P, 0.7))
    print(f'[기준: 스태킹 없음 v3] Val Macro F1  raw {E.f1(vc.label, base_raw):.4f} | 확신cummax {E.f1(vc.label, base_post):.4f}')

    order = pd.read_csv(f'{D}/sample_submission.csv', dtype={'sample_id': 'string'})['sample_id']
    for name, P_train in [('in-sample', P_in), ('OOF', P_oof)]:
        S_tr = stack_features(tr_cap, P_train)
        m2 = fit_stage2(X1_tr, S_tr, tr_cap.label.values)
        Q_va = predict_stage2(m2, X1_va, S_va)
        raw = E.L[Q_va.argmax(1)]
        post = E._per_exp(vc, Q_va, lambda P, t: E.conf_cummax(P, 0.7))
        print(f'\n[스태킹 {name}] Val Macro F1  raw {E.f1(vc.label, raw):.4f} | 확신cummax {E.f1(vc.label, post):.4f}')
        R.print_tables(vc, post, f'Val — 스태킹 {name} + 확신 cummax τ=0.7')
        # Test 제출 파일
        Q_te = predict_stage2(m2, X1_te, S_te)
        lab = pd.Series(E._per_exp(te, Q_te, lambda P, t: E.conf_cummax(P, 0.7)), index=te.sample_id.values)
        out = f'submission_stack_{name.replace("-", "")}.csv'
        pd.DataFrame({'sample_id': order, 'label': lab.loc[order].values}).to_csv(out, index=False)
        print('Test 제출 파일:', out, '| 라벨 분포', lab.value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
