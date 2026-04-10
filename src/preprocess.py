import numpy as np
import pandas as pd

from .config import ALL_CATEGORICAL


def cast_category_types(df):
    """
    Casts categorical columns to pandas 'category' dtype
    (required for XGBoost enable_categorical=True).
    """
    df = df.copy()

    for col in ALL_CATEGORICAL:
        df[col] = df[col].astype("category")

    return df

def remove_expired_patients(df):
    """
    Removing patients that have expired or have been admitted to hospice.
    patients that are expired cannot be readmitted and patients admitted to hospices
    are usually receiving end-of-life treatments or are terminally-ill. In both of these 
    cases patient is unlikely to be readmitted so this coudl cause leakage 
    """
    expired_ids = [11, 13, 14, 19, 20, 21]
    df = df[df["discharge_disposition_id"].isin(expired_ids) == False]
    return df
