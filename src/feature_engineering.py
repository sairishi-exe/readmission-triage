import pandas as pd


def add_utilization_index(df):
    """
    Total prior healthcare visits = inpatient + outpatient + emergency.
    Captures overall which patients are frequent visitors to the
    healthcare facility.
    """
    df = df.copy()
    df["utilization_index"] = (
        df["number_inpatient"]
        + df["number_outpatient"]
        + df["number_emergency"]
    )
    return df


def add_med_change_count(df):
    """
    The point of this function is to assign a number that distinguishes
    b/w types of medication changes occurring for each encounter
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
    available_meds = [c for c in med_cols if c in df.columns]
    med_df = df[available_meds]
    df["med_up_count"] = (med_df == "Up").sum(axis=1)
    df["med_down_count"] = (med_df == "Down").sum(axis=1)
    df["med_steady_count"] = (med_df == "Steady").sum(axis=1)

    return df

def count_repeated_encounters(df):
    """ 
    Some patients identified by patient_nbr appear multiple times
    in the dataset identified by encounter_id. This could be a signal.
    """
    df["encounter_count"] = df.groupby("patient_nbr")["encounter_id"].transform("count")
    return df


