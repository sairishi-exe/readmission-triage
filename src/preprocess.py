import numpy as np
import pandas as pd

from .config import NONE_IS_VALID_COLS, ALL_CATEGORICAL


def clean_data(df):
    """
    Domain-specific NaN handling:
      - A1Cresult, max_glu_serum: NaN -> "None" (test not ordered, not missing)
      - Everything else: left as NaN for XGBoost sparsity-aware splitting

    Casts categorical columns to pandas 'category' dtype
    (required for XGBoost enable_categorical=True).
    """
    df = df.copy()

    for col in NONE_IS_VALID_COLS:
        df[col] = df[col].fillna("None")

    for col in ALL_CATEGORICAL:
        df[col] = df[col].astype("category")

    return df

def remove_expired_patients(df):
    expired_ids = [11, 13, 14, 19, 20, 21]
    df = df[df["discharge_disposition_id"].isin(expired_ids) == False]
    return df
