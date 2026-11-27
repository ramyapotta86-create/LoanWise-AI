"""Responsible AI demographic fairness audit and ablation checks."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.src.data_loader import load_training_data
from ml.src.evaluate import evaluate_on_test
from ml.src.feature_engineering import RAW_FEATURE_COLUMNS, clean_string_series
from ml.src.predict import load_model_artifacts
from ml.src.preprocessing import (
    CATEGORICAL_COLUMNS,
    build_preprocessing_pipeline,
    prepare_features_and_target,
)
from ml.src.train import RANDOM_STATE, TEST_SIZE

SENSITIVE_ATTRIBUTES = ["Gender", "Married"]


def compute_subgroup_fairness(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    attribute: str,
) -> Dict[str, Any]:
    """Compute subgroup approval rates, TPR, FNR, DPD, and EOD for one attribute."""
    if attribute not in X_test.columns:
        raise ValueError(f"Sensitive attribute '{attribute}' not present in test DataFrame.")

    groups = clean_string_series(X_test[attribute]).fillna("<MISSING>")
    eval_df = pd.DataFrame(
        {
            "group": groups.values,
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    )

    subgroups: List[Dict[str, Any]] = []
    pos_rates: List[float] = []
    tprs: List[float] = []

    for group_val, sub in eval_df.groupby("group"):
        n_samples = int(len(sub))
        actual_rate = float(sub["y_true"].mean())
        pred_rate = float(sub["y_pred"].mean())

        positives = sub[sub["y_true"] == 1]
        if len(positives) > 0:
            tpr = float((positives["y_pred"] == 1).mean())
            fnr = float(1.0 - tpr)
        else:
            tpr = None
            fnr = None

        if group_val != "<MISSING>":
            pos_rates.append(pred_rate)
            if tpr is not None:
                tprs.append(tpr)

        subgroups.append(
            {
                "group": str(group_val),
                "sample_count": n_samples,
                "actual_approval_rate": round(actual_rate, 4),
                "predicted_approval_rate": round(pred_rate, 4),
                "true_positive_rate": round(tpr, 4) if tpr is not None else None,
                "false_negative_rate": round(fnr, 4) if fnr is not None else None,
            }
        )

    dpd = round(max(pos_rates) - min(pos_rates), 4) if len(pos_rates) >= 2 else None
    eod = round(max(tprs) - min(tprs), 4) if len(tprs) >= 2 else None

    return {
        "attribute": attribute,
        "subgroups": subgroups,
        "demographic_parity_difference": dpd,
        "equal_opportunity_difference": eod,
    }


def run_fairness_audit(
    df: Optional[pd.DataFrame] = None,
    baseline_pipeline: Optional[Pipeline] = None,
    scale_numeric: Optional[bool] = None,
) -> Dict[str, Any]:
    """Compare Model A (all 11 features) vs Model B (without Gender & Married)."""
    raw_df = df if df is not None else load_training_data()
    X, y = prepare_features_and_target(raw_df)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    if baseline_pipeline is None:
        model_a, metadata = load_model_artifacts()
        is_scaled = any(
            name == "scaler"
            for name, _ in model_a.named_steps["column_transformer"]
            .named_transformers_["num_continuous"]
            .steps
        )
    else:
        model_a = baseline_pipeline
        metadata = {}
        is_scaled = (
            scale_numeric
            if scale_numeric is not None
            else any(
                name == "scaler"
                for name, _ in model_a.named_steps["column_transformer"]
                .named_transformers_["num_continuous"]
                .steps
            )
        )

    # Model A metrics and subgroup checks on held-out test set
    y_pred_a = model_a.predict(X_test)
    metrics_a = evaluate_on_test(model_a, X_test, y_test, X_train=X_dev, y_train=y_dev)
    fairness_a = [
        compute_subgroup_fairness(X_test, y_test, y_pred_a, attr)
        for attr in SENSITIVE_ATTRIBUTES
    ]

    # Model B: Ablated model excluding Gender and Married
    ablated_raw_cols = [c for c in RAW_FEATURE_COLUMNS if c not in SENSITIVE_ATTRIBUTES]
    ablated_cat_cols = [c for c in CATEGORICAL_COLUMNS if c not in SENSITIVE_ATTRIBUTES]
    preproc_b = build_preprocessing_pipeline(
        scale_numeric=is_scaled,
        feature_columns=ablated_raw_cols,
        categorical_columns=ablated_cat_cols,
    )
    classifier_b = clone(model_a.named_steps["classifier"])
    model_b = Pipeline(
        steps=[
            ("feature_engineer", preproc_b.named_steps["feature_engineer"]),
            ("column_transformer", preproc_b.named_steps["column_transformer"]),
            ("classifier", classifier_b),
        ]
    )
    model_b.fit(X_dev[ablated_raw_cols], y_dev)

    y_pred_b = model_b.predict(X_test[ablated_raw_cols])
    metrics_b = evaluate_on_test(
        model_b,
        X_test[ablated_raw_cols],
        y_test,
        X_train=X_dev[ablated_raw_cols],
        y_train=y_dev,
    )
    fairness_b = [
        compute_subgroup_fairness(X_test, y_test, y_pred_b, attr)
        for attr in SENSITIVE_ATTRIBUTES
    ]

    return {
        "model_name": metadata.get("selected_model", type(classifier_b).__name__),
        "model_a_with_demographics": {
            "metrics": metrics_a,
            "subgroup_fairness": fairness_a,
        },
        "model_b_without_demographics": {
            "metrics": metrics_b,
            "subgroup_fairness": fairness_b,
        },
        "limitations": [
            "Historical dataset (~614 rows) may reflect past underwriting biases rather than true repayment capability.",
            "Small held-out test subgroup sizes (e.g., Female applicants) make fairness gaps noisy and must be interpreted carefully.",
            "Removing Gender and Married alone does not guarantee fairness if correlated proxy variables remain.",
        ],
    }


if __name__ == "__main__":
    try:
        report = run_fairness_audit()
        print(f"Fairness Ablation Report ({report['model_name']})")
        for label, key in [
            ("Model A (With Gender & Married)", "model_a_with_demographics"),
            ("Model B (Without Gender & Married)", "model_b_without_demographics"),
        ]:
            m = report[key]["metrics"]
            print(f"\n--- {label} ---")
            print(
                f"Test Accuracy={m['test_accuracy']} | F1={m['test_f1']} | ROC-AUC={m['test_roc_auc']}"
            )
            for fg in report[key]["subgroup_fairness"]:
                print(
                    f"  {fg['attribute']}: DPD={fg['demographic_parity_difference']} | "
                    f"EOD={fg['equal_opportunity_difference']}"
                )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Setup Needed] {exc}")
