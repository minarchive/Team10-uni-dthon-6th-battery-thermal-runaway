from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_link"
FOLDS = ROOT / "data" / "folds.parquet"
NUM_CLASSES = 6


def table() -> pd.DataFrame:
    """train+val(라벨 있음, fold 0..4) + test(label=-1, fold=-1) 전체 표. 행 순서가 모든 feature 캐시의 순서."""
    return pd.read_parquet(FOLDS)
