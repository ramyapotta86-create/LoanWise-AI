"""Model training, stratified CV comparison, selection, and Joblib export."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml.src.data_loader import load_training_data
from ml.src.evaluate import build_comparison_dataframe, evaluate_on_test, run_stratified_cv
from ml.src.explainability import compute_global_shap_importance, create_shap_background
from ml.src.feature_engineering import RAW_FEATURE_COLUMNS, TARGET_MAPPING
from ml.src.preprocessing import build_preprocessing_pipeline, get_transformed_feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"
MODEL_PIPELINE_FILENAME = "model_pipeline.joblib"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
SHAP_BACKGROUND_FILENAME = "shap_background.joblib"
METADATA_FILENAME = "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5


def get_candidate_specs() -> List[Dict[str, Any]]:
    """Return candidate model definitions and bounded CV hyperparameter grids."""
    return [
        {
            "model_name": "LogisticRegression",
            "scale_numeric": True,
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        {
            "model_name": "DecisionTreeClassifier",
            "scale_numeric": False,
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__max_depth": [3, 4, 5, 6],
                "classifier__min_samples_leaf": [5, 10, 15],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        {
            "model_name": "RandomForestClassifier",
            "scale_numeric": False,
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, 6, 8],
                "classifier__min_samples_leaf": [3, 5, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        },
    ]


def build_candidate_pipeline(estimator: Any, scale_numeric: bool) -> Pipeline:
    """Wrap preprocessing steps and classifier into a single leak-free Pipeline."""
    preproc = build_preprocessing_pipeline(scale_numeric=scale_numeric)
    return Pipeline(
        steps=[
            ("feature_engineer", preproc.named_steps["feature_engineer"]),
            ("column_transformer", preproc.named_steps["column_transformer"]),
            ("classifier", estimator),
        ]
    )


def select_best_model(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select winning model by CV F1-score, then CV ROC-AUC, then smaller overfitting gap."""
    return max(
        results,
        key=lambda r: (
            r["cv_f1_mean"],
            r["cv_roc_auc_mean"],
            r["test_f1"],
            -(abs(r.get("overfitting_gap_f1", 0.0))),
        ),
    )


def train_and_evaluate_models(
    df: Optional[pd.DataFrame] = None,
    output_dir: Path = SAVED_MODELS_DIR,
    cv_splits: int = CV_SPLITS,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Run full leak-free training, CV tuning, test evaluation, and artifact saving."""
    from ml.src.preprocessing import prepare_features_and_target

    raw_df = df if df is not None else load_training_data()
    X, y = prepare_features_and_target(raw_df)

    # Lock away 20% held-out test set BEFORE any tuning or fitting
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    all_results: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for spec in get_candidate_specs():
        model_name = spec["model_name"]
        base_pipeline = build_candidate_pipeline(
            estimator=clone(spec["estimator"]),
            scale_numeric=spec["scale_numeric"],
        )

        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=spec["param_grid"],
            cv=cv,
            scoring="f1",
            refit=True,
            n_jobs=1,
        )
        search.fit(X_dev, y_dev)
        best_pipeline: Pipeline = search.best_estimator_

        # Run 5-fold Stratified CV on the tuned pipeline config to get mean ± std for all metrics
        cv_metrics = run_stratified_cv(
            clone(best_pipeline),
            X_dev,
            y_dev,
            n_splits=cv_splits,
            random_state=random_state,
        )
        test_metrics = evaluate_on_test(
            best_pipeline,
            X_test,
            y_test,
            X_train=X_dev,
            y_train=y_dev,
        )

        clean_params = {
            k.replace("classifier__", ""): v for k, v in search.best_params_.items()
        }

        combined: Dict[str, Any] = {
            "model_name": model_name,
            "scale_numeric": spec["scale_numeric"],
            "best_params": clean_params,
            **cv_metrics,
            **test_metrics,
        }
        all_results.append(combined)
        fitted_pipelines[model_name] = best_pipeline

    winner = select_best_model(all_results)
    winning_name = winner["model_name"]
    winning_pipeline = fitted_pipelines[winning_name]

    # Save fitted full pipeline, standalone preprocessor, and SHAP background sample
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_PIPELINE_FILENAME
    preprocessor_path = output_dir / PREPROCESSOR_FILENAME
    shap_bg_path = output_dir / SHAP_BACKGROUND_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    standalone_preprocessor = Pipeline(
        steps=[
            ("feature_engineer", winning_pipeline.named_steps["feature_engineer"]),
            ("column_transformer", winning_pipeline.named_steps["column_transformer"]),
        ]
    )

    shap_background = create_shap_background(winning_pipeline, X_dev)
    global_shap_importance = compute_global_shap_importance(winning_pipeline, shap_background)

    joblib.dump(winning_pipeline, model_path)
    joblib.dump(standalone_preprocessor, preprocessor_path)
    joblib.dump(shap_background, shap_bg_path)

    transformed_features = get_transformed_feature_names(winning_pipeline)
    metadata = {
        "model_version": "v1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selected_model": winning_name,
        "selected_params": winner["best_params"],
        "class_mapping": {
            "Y": TARGET_MAPPING["Y"],
            "N": TARGET_MAPPING["N"],
            "1": "Approved",
            "0": "Rejected",
        },
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "engineered_features": ["TotalIncome"],
        "transformed_feature_names": transformed_features,
        "dataset_summary": {
            "rows_after_dedup_and_target_clean": int(len(X)),
            "dev_rows": int(len(X_dev)),
            "test_rows": int(len(X_test)),
            "class_counts": {
                "Rejected (0)": int((y == 0).sum()),
                "Approved (1)": int((y == 1).sum()),
            },
            "cv_splits": cv_splits,
            "test_size": test_size,
            "random_state": random_state,
        },
        "selected_metrics": {
            "cv_accuracy_mean": winner["cv_accuracy_mean"],
            "cv_f1_mean": winner["cv_f1_mean"],
            "cv_roc_auc_mean": winner["cv_roc_auc_mean"],
            "test_accuracy": winner["test_accuracy"],
            "test_precision": winner["test_precision"],
            "test_recall": winner["test_recall"],
            "test_f1": winner["test_f1"],
            "test_roc_auc": winner["test_roc_auc"],
            "confusion_matrix": winner["confusion_matrix"],
        },
        "model_comparison": all_results,
        "global_feature_importance": global_shap_importance,
        "artifacts": {
            "model_pipeline": _relative_or_str(model_path),
            "preprocessor": _relative_or_str(preprocessor_path),
            "shap_background": _relative_or_str(shap_bg_path),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return {
        "metadata": metadata,
        "comparison_df": build_comparison_dataframe(all_results),
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "shap_background_path": shap_bg_path,
        "metadata_path": metadata_path,
    }


def _relative_or_str(path: Path) -> str:
    """Python 3.8-safe relative path formatting."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    try:
        outcome = train_and_evaluate_models()
        print("\n=== Model Comparison (5-Fold Stratified CV + 20% Held-Out Test) ===")
        print(outcome["comparison_df"].to_string(index=False))
        meta = outcome["metadata"]
        print(f"\nSelected Best Model: {meta['selected_model']}")
        print(f"Best Parameters:     {meta['selected_params']}")
        print(f"Test Confusion Matrix: {meta['selected_metrics']['confusion_matrix']['matrix']}")
        print(f"Saved model pipeline:  {outcome['model_path']}")
        print(f"Saved preprocessor:    {outcome['preprocessor_path']}")
        print(f"Saved metadata JSON:   {outcome['metadata_path']}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Setup Needed] {exc}")
