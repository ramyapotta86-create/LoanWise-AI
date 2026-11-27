"""Model evaluation and stratified cross-validation helpers for LoanWise AI."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def run_stratified_cv(
    pipeline: Pipeline,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Run Stratified K-Fold CV on the development set and return mean/std metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(
        pipeline,
        X_dev,
        y_dev,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1,
    )

    summary: Dict[str, float] = {}
    for metric in scoring:
        test_vals = scores[f"test_{metric}"]
        summary[f"cv_{metric}_mean"] = round(float(np.mean(test_vals)), 4)
        summary[f"cv_{metric}_std"] = round(float(np.std(test_vals)), 4)

    summary["cv_train_f1_mean"] = round(float(np.mean(scores["train_f1"])), 4)
    summary["cv_train_accuracy_mean"] = round(float(np.mean(scores["train_accuracy"])), 4)
    return summary


def evaluate_on_test(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Evaluate a fitted pipeline once on the held-out test set."""
    y_pred = fitted_pipeline.predict(X_test)
    y_proba: Optional[np.ndarray] = None
    if hasattr(fitted_pipeline, "predict_proba"):
        y_proba = fitted_pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Rejected (0)", "Approved (1)"],
        output_dict=True,
        zero_division=0,
    )

    metrics: Dict[str, Any] = {
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4) if y_proba is not None else None,
        "confusion_matrix": {
            "labels": ["Rejected (0)", "Approved (1)"],
            "matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "per_class_report": report,
    }

    if X_train is not None and y_train is not None:
        y_train_pred = fitted_pipeline.predict(X_train)
        train_acc = float(accuracy_score(y_train, y_train_pred))
        train_f1 = float(f1_score(y_train, y_train_pred, zero_division=0))
        metrics["train_accuracy"] = round(train_acc, 4)
        metrics["train_f1"] = round(train_f1, 4)
        metrics["overfitting_gap_f1"] = round(train_f1 - metrics["test_f1"], 4)

    return metrics


def build_comparison_dataframe(model_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format model comparison dictionaries into a clean summary DataFrame."""
    rows = []
    for item in model_results:
        rows.append(
            {
                "model": item["model_name"],
                "cv_accuracy": f"{item['cv_accuracy_mean']:.4f} ± {item['cv_accuracy_std']:.4f}",
                "cv_f1": f"{item['cv_f1_mean']:.4f} ± {item['cv_f1_std']:.4f}",
                "cv_roc_auc": f"{item['cv_roc_auc_mean']:.4f} ± {item['cv_roc_auc_std']:.4f}",
                "test_accuracy": item["test_accuracy"],
                "test_precision": item["test_precision"],
                "test_recall": item["test_recall"],
                "test_f1": item["test_f1"],
                "test_roc_auc": item["test_roc_auc"],
                "overfitting_gap_f1": item.get("overfitting_gap_f1"),
            }
        )
    return pd.DataFrame(rows)
