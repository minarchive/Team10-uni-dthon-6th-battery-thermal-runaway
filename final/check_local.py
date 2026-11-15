"""제출 전 로컬 점검(운영진 점검과 같은 방식): test.py 를 여러 번 돌려 결과가 같은지 확인한다.
 1) 같은 입력 재실행  2) 행 순서 섞기  3) 실험 나눠 넣기  4) 실험마다 앞 60%만 넣기(미래 행 제거, 남은 행의 예측이 같아야 함)
사용: python check_local.py --data_dir /path/to/battery_data_light"""
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', required=True); a = ap.parse_args(); D = Path(a.data_dir); here = Path(__file__).parent; tmp = Path(tempfile.mkdtemp())
def run(df, tag):
    inp = tmp / f'{tag}.csv'; out = tmp / f'{tag}_out.csv'; df.to_csv(inp, index=False)
    subprocess.run([sys.executable, str(here / 'test.py'), '--input', str(inp), '--image_root', str(D / 'test'), '--output', str(out), '--weights', str(here / 'weights')], check=True, cwd=here, capture_output=True)
    return pd.read_csv(out).set_index('sample_id').label
te = pd.read_csv(D / 'test' / 'test.csv'); full = run(te, 'full'); ok = lambda c: '✓' if c else '✗'
r = run(te, 'again'); print(ok((full.loc[r.index] == r).all()), '같은 입력 다시 실행 (다른 행', int((full.loc[r.index] != r).sum()), ')')
r = run(te.sample(frac=1, random_state=0), 'shuffle'); print(ok((full.loc[r.index] == r).all()), '행 순서 섞기 (다른 행', int((full.loc[r.index] != r).sum()), ')')
ex = te.experiment_id.unique(); ch = np.array_split(ex, 3); r = pd.concat([run(te[te.experiment_id.isin(c)], f'split{i}') for i, c in enumerate(ch)]); print(ok((full.loc[r.index] == r).all()), '실험 나눠 넣기 (다른 행', int((full.loc[r.index] != r).sum()), ')')
pre = te.sort_values(['experiment_id', 'relative_time_s'], kind='stable').groupby('experiment_id', group_keys=False).apply(lambda g: g.iloc[:max(1, int(len(g) * 0.6))]); r = run(pre, 'prefix60'); print(ok((full.loc[r.index] == r).all()), '앞 60%만 넣기 (다른 행', int((full.loc[r.index] != r).sum()), ')')
