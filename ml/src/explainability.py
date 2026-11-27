"""SHAP explainability utilities and one-hot feature aggregation for LoanWise AI."""

from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml.src.feature_engineering import RAW_FEATURE_COLUMNS
from ml.src.preprocessing import (
    get_transformed_feature_names,
    map_transformed_name_to_raw,
)

CAUSATION_NOTICE = (
    "SHAP values show how each input influenced this statistical model's prediction "
    "relative to the training dataset baseline; they do not prove real-world banking causation."
)


def extract_preprocessor(model_pipeline: Pipeline) -> Pipeline:
    """Return the preprocessing sub-pipeline (feature engineering + column transformer)."""
    return Pipeline(
        steps=[
            ("feature_engineer", model_pipeline.named_steps["feature_engineer"]),
            ("column_transformer", model_pipeline.named_steps["column_transformer"]),
        ]
    )


def create_shap_background(
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    max_samples: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """Create a compact transformed background dataset for SHAP explainers."""
    preprocessor = extract_preprocessor(model_pipeline)
    sample_df = X_train
    if len(X_train) > max_samples:
        sample_df = X_train.sample(n=max_samples, random_state=random_state)
    return np.asarray(preprocessor.transform(sample_df), dtype=float)


def build_shap_explainer(
    model_pipeline: Pipeline,
    background_transformed: Optional[np.ndarray] = None,
) -> Tuple[Any, str]:
    """Select and initialize the appropriate SHAP explainer for the winning model."""
    classifier = model_pipeline.named_steps["classifier"]

    if isinstance(classifier, (DecisionTreeClassifier, RandomForestClassifier)):
        return shap.TreeExplainer(classifier), "TreeExplainer"

    if isinstance(classifier, LogisticRegression):
        if background_transformed is None:
            raise ValueError(
                "background_transformed is required to initialize shap.LinearExplainer "
                "for LogisticRegression."
            )
        return shap.LinearExplainer(classifier, background_transformed), "LinearExplainer"

    if background_transformed is None:
        raise ValueError("background_transformed is required for KernelExplainer fallback.")
    return shap.KernelExplainer(classifier.predict_proba, background_transformed), "KernelExplainer"


def _extract_positive_class_shap(
    shap_values: Any,
    expected_value: Any,
) -> Tuple[np.ndarray, float]:
    """Normalize SHAP output shapes across classifiers and SHAP versions for class 1."""
    if isinstance(shap_values, list):
        class_idx = 1 if len(shap_values) > 1 else 0
        row_shap = np.asarray(shap_values[class_idx])[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            class_idx = 1 if arr.shape[2] > 1 else 0
            row_shap = arr[0, :, class_idx]
        elif arr.ndim == 2:
            row_shap = arr[0, :]
        else:
            row_shap = arr.ravel()

    if isinstance(expected_value, (list, tuple, np.ndarray)) and np.size(expected_value) > 1:
        base_val = float(np.ravel(expected_value)[1])
    else:
        base_val = float(np.ravel(expected_value)[0])

    return row_shap.astype(float), base_val


def _extract_matrix_positive_class_shap(shap_values: Any) -> np.ndarray:
    """Normalize batch SHAP values to a 2D array (n_samples, n_features) for class 1."""
    if isinstance(shap_values, list):
        class_idx = 1 if len(shap_values) > 1 else 0
        return np.asarray(shap_values[class_idx], dtype=float)
    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        class_idx = 1 if arr.shape[2] > 1 else 0
        return arr[:, :, class_idx]
    return arr


def explain_single_application(
    model_pipeline: Pipeline,
    application_df: pd.DataFrame,
    background_transformed: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute aggregated feature-level SHAP explanations for a single applicant row."""
    if len(application_df) != 1:
        raise ValueError("explain_single_application expects a single-row DataFrame.")

    preprocessor = extract_preprocessor(model_pipeline)
    engineered_df = model_pipeline.named_steps["feature_engineer"].transform(application_df)
    transformed_row = np.asarray(preprocessor.transform(application_df), dtype=float)
    transformed_names = get_transformed_feature_names(model_pipeline)

    explainer, explainer_type = build_shap_explainer(
        model_pipeline,
        background_transformed=background_transformed,
    )
    raw_shap = explainer.shap_values(transformed_row)
    row_shap, base_value = _extract_positive_class_shap(raw_shap, explainer.expected_value)

    # Aggregate one-hot encoded categorical SHAP contributions back to raw feature names
    grouped_shap: Dict[str, float] = {}
    for feat_name, shap_val in zip(transformed_names, row_shap):
        raw_name = map_transformed_name_to_raw(feat_name)
        grouped_shap[raw_name] = grouped_shap.get(raw_name, 0.0) + float(shap_val)

    raw_row = application_df.iloc[0]
    contributions: List[Dict[str, Any]] = []
    for feat, val in grouped_shap.items():
        if feat in RAW_FEATURE_COLUMNS:
            input_val = str(raw_row[feat])
        elif feat in engineered_df.columns:
            input_val = str(engineered_df.iloc[0][feat])
        else:
            input_val = "N/A"

        contributions.append(
            {
                "feature": feat,
                "input_value": input_val,
                "shap_value": round(float(val), 6),
                "direction": "positive" if val >= 0 else "negative",
            }
        )

    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

    proba = float(model_pipeline.predict_proba(application_df)[0, 1])
    prediction = "Approved" if proba >= 0.50 else "Rejected"

    summary_points = []
    for item in contributions[:3]:
        effect = "increased" if item["shap_value"] >= 0 else "decreased"
        summary_points.append(
            f"{item['feature']} = {item['input_value']} {effect} the model's "
            f"approval score (SHAP: {item['shap_value']:+.4f})."
        )

    return {
        "prediction": prediction,
        "approval_probability": round(proba, 4),
        "explainer_type": explainer_type,
        "base_value": round(base_value, 6),
        "contributions": contributions,
        "summary_points": summary_points,
        "causation_notice": CAUSATION_NOTICE,
    }


def compute_global_shap_importance(
    model_pipeline: Pipeline,
    background_transformed: np.ndarray,
) -> List[Dict[str, Any]]:
    """Compute mean absolute SHAP importance aggregated to original feature names."""
    transformed_names = get_transformed_feature_names(model_pipeline)
    explainer, _ = build_shap_explainer(
        model_pipeline,
        background_transformed=background_transformed,
    )
    shap_matrix = _extract_matrix_positive_class_shap(
        explainer.shap_values(background_transformed)
    )

    # Sum transformed SHAP columns per raw feature for each sample before averaging |SHAP|
    raw_order: List[str] = []
    col_indices: Dict[str, List[int]] = {}
    for idx, feat_name in enumerate(transformed_names):
        raw_name = map_transformed_name_to_raw(feat_name)
        if raw_name not in col_indices:
            col_indices[raw_name] = []
            raw_order.append(raw_name)
        col_indices[raw_name].append(idx)

    importance_list: List[Dict[str, Any]] = []
    for raw_name in raw_order:
        grouped_vals = shap_matrix[:, col_indices[raw_name]].sum(axis=1)
        mean_abs = float(np.mean(np.abs(grouped_vals)))
        importance_list.append(
            {"feature": raw_name, "mean_abs_shap": round(mean_abs, 6)}
        )

    importance_list.sort(key=lambda item: item["mean_abs_shap"], reverse=True)
    return importance_list


def load_shap_background(background_path: str) -> np.ndarray:
    """Load saved SHAP background array from Joblib."""
    return np.asarray(joblib.load(background_path), dtype=float)
