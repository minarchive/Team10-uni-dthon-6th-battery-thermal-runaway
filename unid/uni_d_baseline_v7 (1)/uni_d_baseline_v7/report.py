"""bat_class × 라벨 성능표 (데이터 분포표와 같은 형태).

bat_class: 1=원통형, 2=파우치형, 3=각형
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = [1, 2, 3, 4, 5, 6]
BAT = {1: '1 원통형', 2: '2 파우치형', 3: '3 각형'}


def _grid(frame, fn):
    rows = {}
    for b in [1, 2, 3, '합계']:
        f = frame if b == '합계' else frame[frame.bat_class == b]
        rows[BAT.get(b, b)] = fn(f)
    return pd.DataFrame(rows).T


def label_tables(frame: pd.DataFrame, pred) -> dict:
    """frame: label, bat_class 컬럼 포함 / pred: frame과 같은 순서의 예측 라벨."""
    f = frame[['label', 'bat_class']].copy()
    f['pred'] = np.asarray(pred)
    f['ok'] = f.label == f.pred
    cols = [f'L{c}' for c in LABELS]

    def count(g):  # 정답 수 / 전체 수
        s = {f'L{c}': f'{int(g.ok[g.label == c].sum()):,} / {int((g.label == c).sum()):,}' for c in LABELS}
        s['합계'] = f'{int(g.ok.sum()):,} / {len(g):,}'
        return s

    def recall(g):
        s = {f'L{c}': (g.ok[g.label == c].mean() * 100 if (g.label == c).any() else np.nan) for c in LABELS}
        s['합계'] = g.ok.mean() * 100
        return s

    def f1(g):
        s = dict(zip(cols, f1_score(g.label, g.pred, labels=LABELS, average=None, zero_division=0) * 100))
        present = [c for c in LABELS if (g.label == c).any()]
        s['Macro'] = f1_score(g.label, g.pred, labels=present, average='macro', zero_division=0) * 100
        return s

    return {
        '정답 수 / 샘플 수': _grid(f, count),
        '재현율 % (해당 라벨 중 맞춘 비율)': _grid(f, recall).round(1),
        'F1 % (Macro는 해당 bat_class에 있는 라벨만 평균)': _grid(f, f1).round(1),
    }


def print_tables(frame, pred, title):
    print(f'\n######## {title} ########')
    for name, t in label_tables(frame, pred).items():
        print(f'\n[{name}]')
        print(t.to_string())


def to_markdown(frame, pred, title) -> str:
    out = [f'### {title}']
    for name, t in label_tables(frame, pred).items():
        out += [f'\n**{name}**\n', t.to_markdown()]
    return '\n'.join(out)
