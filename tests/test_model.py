"""Unit tests for model training, comparison, Joblib persistence, inference, and SHAP."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.explainability import explain_single_application
from ml.src.fairness import run_fairness_audit
from ml.src.feature_engineering import RAW_FEATURE_COLUMNS
from ml.src.predict import load_model_artifacts, predict_loan_application
from ml.src.train import (
    METADATA_FILENAME,
    MODEL_PIPELINE_FILENAME,
    PREPROCESSOR_FILENAME,
    SHAP_BACKGROUND_FILENAME,
    train_and_evaluate_models,
)


def _synthetic_training_df(n_rows: int = 80) -> pd.DataFrame:
    """Create a small synthetic DataFrame solely to test training/saving mechanics."""
    rng = np.random.default_rng(42)
    rows = []
    for idx in range(n_rows):
        credit = float(1 if idx % 3 != 0 else 0)
        income = int(rng.integers(2500, 10000))
        co_income = int(rng.choice([0, 1200, 2500]))
        loan_amt = float(rng.integers(80, 260))
        status = "Y" if (credit == 1.0 and (income + co_income) > 4500) else "N"
        rows.append(
            {
                "Loan_ID": f"LP{1000 + idx}",
                "Gender": "Male" if idx % 2 == 0 else "Female",
                "Married": "Yes" if idx % 3 == 0 else "No",
                "Dependents": ["0", "1", "2", "3+"][idx % 4],
                "Education": "Graduate" if idx % 4 != 0 else "Not Graduate",
                "Self_Employed": "No" if idx % 5 != 0 else "Yes",
                "ApplicantIncome": income,
                "CoapplicantIncome": co_income,
                "LoanAmount": np.nan if idx % 15 == 0 else loan_amt,
                "Loan_Amount_Term": 360.0,
                "Credit_History": np.nan if idx % 20 == 0 else credit,
                "Property_Area": ["Urban", "Semiurban", "Rural"][idx % 3],
                "Loan_Status": status,
            }
        )
    return pd.DataFrame(rows)


def test_train_save_predict_and_shap_roundtrip(tmp_path: Path) -> None:
    df = _synthetic_training_df(n_rows=80)
    train_and_evaluate_models(df=df, output_dir=tmp_path, cv_splits=3)

    model_path = tmp_path / MODEL_PIPELINE_FILENAME
    preproc_path = tmp_path / PREPROCESSOR_FILENAME
    shap_bg_path = tmp_path / SHAP_BACKGROUND_FILENAME
    meta_path = tmp_path / METADATA_FILENAME

    assert model_path.exists()
    assert preproc_path.exists()
    assert shap_bg_path.exists()
    assert meta_path.exists()

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    benchmarked = {item["model_name"] for item in metadata["model_comparison"]}
    assert benchmarked == {
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
    }
    assert metadata["selected_model"] in benchmarked

    # Verify Joblib reload via predict.py helper
    loaded_model, loaded_meta = load_model_artifacts(models_dir=tmp_path)
    assert loaded_meta["selected_model"] == metadata["selected_model"]

    sample_payload = df[RAW_FEATURE_COLUMNS].iloc[0].to_dict()
    prediction_result = predict_loan_application(loaded_model, sample_payload)
    assert prediction_result["prediction"] in ("Approved", "Rejected")
    assert prediction_result["predicted_label"] in (0, 1)
    assert 0.0 <= prediction_result["approval_probability"] <= 1.0
    assert "TotalIncome" in prediction_result["engineered_features"]

    # Verify standalone preprocessor reload
    loaded_preproc = joblib.load(preproc_path)
    sample_df = df[RAW_FEATURE_COLUMNS].iloc[[0]].copy()
    transformed = loaded_preproc.transform(sample_df)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()

    # Verify SHAP explanation & one-hot aggregation
    shap_bg = joblib.load(shap_bg_path)
    explanation = explain_single_application(
        loaded_model,
        sample_df,
        background_transformed=shap_bg,
    )
    assert explanation["explainer_type"] in ("TreeExplainer", "LinearExplainer", "KernelExplainer")
    assert len(explanation["contributions"]) >= len(RAW_FEATURE_COLUMNS)
    assert "causation_notice" in explanation

    # Verify Responsible AI demographic ablation check
    fairness_report = run_fairness_audit(df=df, baseline_pipeline=loaded_model)
    assert "model_a_with_demographics" in fairness_report
    assert "model_b_without_demographics" in fairness_report
    assert len(fairness_report["model_a_with_demographics"]["subgroup_fairness"]) == 2
