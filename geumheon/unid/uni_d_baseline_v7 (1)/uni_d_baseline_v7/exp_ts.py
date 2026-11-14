"""시계열 피처 실험: 피처 구성 / 클래스 가중치 / 다운샘플 증강 / 단조 후처리 비교.

평가 방식 (Test 조건 재현: 500행 초과 실험은 500행으로 균등 다운샘플)
- CV : Train 실험 단위 GroupKFold(5), 검증 fold에 500행 캡 적용
- Val: 제공 Validation(학습에 미사용)에 500행 캡 적용. 참고로 캡 없는 원본 점수도 기록
"""
import argparse, json, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import ts_model as M


def f1(y, p):
    return f1_score(y, p, labels=M.LABELS, average='macro', zero_division=0)


def load(d, split):
    f = pd.read_csv(Path(d) / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    return f.reset_index(drop=True)


def make_model(cw, seed=M.SEED):
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
        class_weight=cw, random_state=seed, n_jobs=18, verbose=-1,
    )


def train_set(frame, aug):
    """aug=True면 긴 실험을 500행으로 줄인 복사본을 추가(피처는 줄인 데이터로 다시 계산)."""
    Xs, ys = [M.build_features(frame)], [frame['label'].to_numpy()]
    if aug:
        long_ids = frame.groupby('experiment_id').size().loc[lambda s: s > M.MAX_ROWS].index
        sub = M.cap_rows(frame[frame.experiment_id.isin(long_ids)])
        if len(sub):
            Xs.append(M.build_features(sub)); ys.append(sub['label'].to_numpy())
    return pd.concat(Xs), np.concatenate(ys)


def evaluate(model, frame, cols, cap=True):
    sub = M.cap_rows(frame) if cap else frame
    p = model.predict(M.build_features(sub)[cols])
    return f1(sub['label'], p), f1(sub['label'], M.monotone(p, sub)), sub, p


SENSOR4 = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p']
FEATSETS = {
    # 베이스라인 센서(meas_a 제외)로 만든 시계열 피처만
    'sensor4': lambda c: [x for x in c if any(x == s or x.startswith(s + '_') for s in SENSOR4)],
    'all': lambda c: list(c),
    'all-time-meta': lambda c: [x for x in c if x != 'relative_time_s' and x not in M.META],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='/home/ubuntu/data/battery_data_light')
    ap.add_argument('--out', default='exp_ts_results.json')
    args = ap.parse_args()
    tr, va = load(args.data_dir, 'train'), load(args.data_dir, 'validation')
    all_cols = list(M.build_features(va.head(50)).columns)
    print('long exps (>500 rows): train', (tr.groupby('experiment_id').size() > 500).sum(),
          '/ val', (va.groupby('experiment_id').size() > 500).sum(), flush=True)

    configs = [
        ('sensor4', None, False), ('all', None, False), ('all', 'balanced', False),
        ('all-time-meta', 'balanced', False), ('all', 'balanced', True),
    ]
    results = []
    for fs, cw, aug in configs:
        t0 = time.time()
        cols = FEATSETS[fs](all_cols)
        cv_raw, cv_mono = [], []
        for a, b in GroupKFold(5).split(tr, groups=tr.experiment_id):
            X, y = train_set(tr.iloc[a].reset_index(drop=True), aug)
            m = make_model(cw).fit(X[cols], y)
            r, mo, _, _ = evaluate(m, tr.iloc[b].reset_index(drop=True), cols)
            cv_raw.append(r); cv_mono.append(mo)
        X, y = train_set(tr, aug)
        m = make_model(cw).fit(X[cols], y)
        v_r, v_m, vsub, vp = evaluate(m, va, cols, cap=True)
        vo_r, _, _, _ = evaluate(m, va, cols, cap=False)
        per_class = f1_score(vsub['label'], M.monotone(vp, vsub), labels=M.LABELS, average=None, zero_division=0)
        row = dict(features=fs, n_feat=len(cols), class_weight=cw or 'none', aug_cap=aug,
                   cv=np.mean(cv_raw), cv_std=np.std(cv_raw), cv_mono=np.mean(cv_mono),
                   val=v_r, val_mono=v_m, val_nocap=vo_r,
                   val_mono_per_class=[round(float(x), 3) for x in per_class], sec=round(time.time() - t0))
        results.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1, default=float))


if __name__ == '__main__':
    main()
