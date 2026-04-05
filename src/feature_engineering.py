import pandas as pd


def add_utilization_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total prior healthcare visits = inpatient + outpatient + emergency.
    Captures overall "frequent flyer" status in one feature.
    """
    df = df.copy()
    df["utilization_index"] = (
        df["number_inpatient"]
        + df["number_outpatient"]
        + df["number_emergency"]
    )
    return df


def add_med_change_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many medications were changed during this encounter.
    Any value other than "No" counts as a change.
    More granular than the binary 'change' column — captures treatment intensity.
    """
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    df = df.copy()
    available = [c for c in med_cols if c in df.columns]
    df["med_change_count"] = df[available].apply(
        lambda row: (row.astype(str) != "No").sum(), axis=1
    )
    return df


