"""CSV loaders.

The three files in `data/raw/` are read as provided. No cleaning step is applied or
needed: the data contains no missing values, duplicate rows, constant columns or
infinities, and every derived feature is computed in memory at run time.
"""
import pandas as pd

from config import SAMPLE_SUB_CSV, TEST_CSV, TRAIN_CSV


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_SUB_CSV)
