import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold


def evaluate_model(model, X, y, threshold=0.5):
    """Evaluate a fitted model. Prints AUCPR, AUROC, precision, recall, confusion matrix."""
    y_proba = model.predict_proba(X)[:, 1] # assigns probability scores to samples
    y_pred = (y_proba >= threshold).astype(int) # uses a threshold to classify based on prob score

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel().tolist()

    aucpr = average_precision_score(y, y_proba)
    auroc = roc_auc_score(y, y_proba)
    precision = precision_score(y, y_pred) # raw scores using 0.5 as decision threshold
    recall = recall_score(y, y_pred)

    print(f"AUCPR:     {aucpr:.4f}")
    print(f"AUROC:     {auroc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Confusion: TN={tn}  FP={fp}  FN={fn}  TP={tp}")


def cross_validate(X, y, params, n_splits=5, random_state=42):
    """Stratified k-fold CV. Prints per-fold AUCPR/AUROC and the mean."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucpr_scores = []
    auroc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()

        model = xgb.XGBClassifier(**params, scale_pos_weight=neg / pos)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        y_proba = model.predict_proba(X_val)[:, 1]
        aucpr = average_precision_score(y_val, y_proba)
        auroc = roc_auc_score(y_val, y_proba)
        aucpr_scores.append(aucpr)
        auroc_scores.append(auroc)

        print(f"Fold {fold}: AUCPR={aucpr:.4f}  AUROC={auroc:.4f}")

    print(f"\nMean AUCPR: {np.mean(aucpr_scores):.4f}")
    print(f"Mean AUROC: {np.mean(auroc_scores):.4f}")
