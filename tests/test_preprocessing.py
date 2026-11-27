"""Unit tests for feature engineering and leak-free preprocessing."""

import numpy as np
import pandas as pd
import pytest

from ml.src.feature_engineering import RAW_FEATURE_COLUMNS, LoanFeatureEngineer
from ml.src.preprocessing import (
    build_preprocessing_pipeline,
    get_transformed_feature_names,
    map_transformed_name_to_raw,
    prepare_features_and_target,
)


def _sample_raw_df() -> pd.DataFrame:
    """Create a small synthetic DataFrame strictly for unit-testing transformer logic."""
    return pd.DataFrame(
        [
            {
                "Loan_ID": "LP0001",
                "Gender": " Male ",
                "Married": "Yes",
                "Dependents": "3+",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 1500,
                "LoanAmount": 120.0,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 1.0,
                "Property_Area": "Urban",
                "Loan_Status": "Y",
            },
            {
                "Loan_ID": "LP0002",
                "Gender": None,
                "Married": "No",
                "Dependents": "0",
                "Education": "Not Graduate",
                "Self_Employed": "Yes",
                "ApplicantIncome": 3000,
                "CoapplicantIncome": 0,
                "LoanAmount": np.nan,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 0.0,
                "Property_Area": "Rural",
                "Loan_Status": "N",
            },
            {
                "Loan_ID": "LP0003",
                "Gender": "Female",
                "Married": "Yes",
                "Dependents": "2",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 4000,
                "CoapplicantIncome": 2000,
                "LoanAmount": 150.0,
                "Loan_Amount_Term": 180.0,
                "Credit_History": np.nan,
                "Property_Area": "Semiurban",
                "Loan_Status": "Y",
            },
        ]
    )


def test_prepare_features_excludes_loan_id_and_maps_target() -> None:
    df = _sample_raw_df()
    X, y = prepare_features_and_target(df)

    assert "Loan_ID" not in X.columns
    assert "Loan_Status" not in X.columns
    assert X.columns.tolist() == RAW_FEATURE_COLUMNS
    assert y.tolist() == [1, 0, 1]


def test_feature_engineer_cleans_strings_dependents_and_total_income() -> None:
    df = _sample_raw_df()
    X, _ = prepare_features_and_target(df)

    engineer = LoanFeatureEngineer()
    transformed = engineer.fit_transform(X)

    assert transformed.loc[0, "Gender"] == "Male"
    assert pd.isna(transformed.loc[1, "Gender"])
    assert transformed.loc[0, "Dependents"] == 3
    assert transformed["TotalIncome"].tolist() == [6500, 3000, 6000]


def test_preprocessing_pipeline_imputes_without_dropping_rows_and_handles_unknown() -> None:
    df = _sample_raw_df()
    X, y = prepare_features_and_target(df)

    pipeline = build_preprocessing_pipeline(scale_numeric=True)
    X_proc = pipeline.fit_transform(X, y)

    assert X_proc.shape[0] == 3
    assert not np.isnan(X_proc).any()

    # Unseen category value during inference should not raise
    unseen = X.iloc[[0]].copy()
    unseen["Property_Area"] = "Coastal"
    unseen_proc = pipeline.transform(unseen)
    assert unseen_proc.shape == (1, X_proc.shape[1])
    assert not np.isnan(unseen_proc).any()


def test_transformed_feature_names_map_to_raw_features() -> None:
    df = _sample_raw_df()
    X, y = prepare_features_and_target(df)
    pipeline = build_preprocessing_pipeline(scale_numeric=False)
    pipeline.fit(X, y)

    names = get_transformed_feature_names(pipeline)
    assert len(names) > len(RAW_FEATURE_COLUMNS)
    mapped = {map_transformed_name_to_raw(name) for name in names}
    assert "Property_Area" in mapped
    assert "TotalIncome" in mapped
    assert "Credit_History" in mapped
