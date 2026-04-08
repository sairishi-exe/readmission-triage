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
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Heatmap")
    plt.show()


def mutual_info_ranking(df, random_state=42):
    """
    Rank features by how much they tell you about the target (readmission).

    Mutual Information (MI) measures dependency between a feature and the target.
    MI = 0 means the feature tells you nothing. Higher = more useful.
    Unlike Pearson correlation, MI catches non-linear relationships too.
    """
    from sklearn.feature_selection import mutual_info_classif
    from .config import ALL_FEATURES, LABEL_COL, ALL_CATEGORICAL

    X = df[[c for c in ALL_FEATURES if c in list(df.columns)]].copy()
    y = (df[LABEL_COL] == "<30").astype(int)

    # Boolean mask for mutual_info_classif:
    # The is_discrete param uses an array of bools to tell apart which cols are categorical
    is_categorical = (X.dtypes == "category").to_list()

    for col in X[ALL_CATEGORICAL].columns:
        X[col] = X[col].cat.codes

    mi_scores = mutual_info_classif(X, y, discrete_features=is_categorical, random_state=random_state)
    score_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    print(score_results.round(4).to_string())
    return score_results


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
