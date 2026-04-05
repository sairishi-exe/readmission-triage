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
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    print(f"AUCPR:     {average_precision_score(y, y_proba):.4f}")
    print(f"AUROC:     {roc_auc_score(y, y_proba):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall:    {recall_score(y, y_pred):.4f}")
    print(f"Confusion: TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")


def cross_validate(X, y, params, n_splits=5, random_state=42):
    """Stratified k-fold CV. Prints per-fold AUCPR/AUROC and the mean."""
    cv_params = params.copy()
    es_rounds = cv_params.pop("early_stopping_rounds")
    n_est = cv_params.pop("n_estimators")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucpr_scores = []
    auroc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()

        cv_model = xgb.XGBClassifier(
            **cv_params,
            scale_pos_weight=neg / pos,
            n_estimators=n_est,
            early_stopping_rounds=es_rounds,
        )
        cv_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=0)

        y_proba = cv_model.predict_proba(X_vl)[:, 1]
        ap = average_precision_score(y_vl, y_proba)
        auc = roc_auc_score(y_vl, y_proba)
        aucpr_scores.append(ap)
        auroc_scores.append(auc)
        print(f"Fold {fold}: AUCPR={ap:.4f}  AUROC={auc:.4f}")

    print(f"\nMean AUCPR: {np.mean(aucpr_scores):.4f}")
    print(f"Mean AUROC: {np.mean(auroc_scores):.4f}")
