import pandas as pd
import numpy as np


def missing_data_stats(df):
    """
    Print missing % for every column. Sorted highest first.
    Columns with 0.0 confirm that clean_data() handled them.
    """
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print((df.isnull().mean() * 100).round(1).sort_values(ascending=False).to_string())


def check_duplicates(df, id_col="patient_nbr"):
    """
    Check if the same patient appears multiple times.
    Important because repeated patients create correlated rows.
    """
    print(f"Unique {id_col}: {df[id_col].nunique():,} / {len(df):,}")


def target_distribution(df, col="readmitted"):
    """
    Show the 3-class target breakdown (NO, >30, <30)
    and the binary positive rate after we collapse it to <30 vs rest.
    """
    print(df[col].value_counts())
    print(f"\n<30 rate: {(df[col] == '<30').mean():.1%}")


def numeric_summary(df, cols):
    """
    Standard statistics for numeric features.
    describe() gives count, mean, std, min, quartiles, max.
    Skewness tells you how lopsided the distribution is:
      0 = symmetric, positive = long right tail, negative = long left tail."""
    print(df[cols].describe().T.round(2).to_string())
    print(f"\nSkewness:\n{df[cols].skew().round(2).to_string()}")


def categorical_summary(df, cols):
    """
    Value counts for each categorical feature.
    Also scans ALL columns in df for zero-variance (only 1 unique value)
    using df[c].nunique() <= 1
    — these are useless and should be dropped.
    """
    zero_var = [c for c in df.columns if df[c].nunique() <= 1]
    if zero_var:
        print(f"Zero-variance (drop): {zero_var}\n")
    for col in cols:
        print(f"\n{col} ({df[col].nunique()} unique)")
        print(df[col].value_counts(dropna=False).to_string())


def feature_correlations(df, cols):
    """
    Pearson correlation heatmap for numeric features.
    Looking for pairs with |r| > 0.9 — those are redundant
    and one can be dropped without losing information.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def mutual_info_ranking(df, random_state=42):
    """
    Rank features by how much they tell you about the target (readmission).

    Mutual Information (MI) measures dependency between a feature and the target.
    MI = 0 means the feature tells you nothing. Higher = more useful.
    Unlike Pearson correlation, MI catches non-linear relationships too.
    """
    # Why OrdinalEncoder instead of get_dummies:
    #   get_dummies turns 'race' into 6 binary columns (race_Caucasian, race_AfricanAmerican, ...).
    #   MI would then score each binary column separately, fragmenting the signal.
    #   OrdinalEncoder keeps 'race' as ONE column (Caucasian=0, AfricanAmerican=1, ...)
    #   so MI scores the full feature as a whole.

    # Why is_categorical mask:
    #   MI computes scores differently depending on the data type.
    #   For numbers like time_in_hospital, it uses nearest-neighbor distances (KSG estimator).
    #   For categories like race (even though encoded as 0,1,2), distances are meaningless —
    #   it instead counts how often each category co-occurs with each target value.
    #   The mask tells MI which method to use for which column.
    
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import OrdinalEncoder
    from .config import ALL_FEATURES, LABEL_COL, ALL_CATEGORICAL

    X = df[[c for c in ALL_FEATURES if c in df.columns]]
    y = (df[LABEL_COL] == "<30").astype(int)

    cat_cols = [c for c in ALL_CATEGORICAL if c in X.columns]

    X_enc = X.copy()
    for c in cat_cols:
        X_enc[c] = X_enc[c].astype(str) # OrdinalEncoder only works with strings for categorical columns
    X_enc[cat_cols] = OrdinalEncoder().fit_transform(X_enc[cat_cols])

    is_categorical = [c in cat_cols for c in X_enc.columns]

    mi = mutual_info_classif(X_enc, y, discrete_features=is_categorical, random_state=random_state)
    scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print(scores.round(4).to_string())
    return scores


def bivariate_target(df, target_col, numeric_cols, categorical_cols):
    """
    Compare features against the raw 3-class target (NO, >30, <30).
    Numeric: mean readmission per target class.
    Categorical: readmission rate (<30d) per category value.
    """
    print("=== Numeric means by target ===")
    print(df.groupby(target_col)[numeric_cols].mean().round(2).T.to_string())

    binary = (df[target_col] == "<30").astype(int)
    print("\n=== Readmission rate by category ===")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(binary.groupby(df[col]).mean().sort_values(ascending=False).to_string())
