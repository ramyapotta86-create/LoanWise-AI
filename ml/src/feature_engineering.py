"""Custom stateless feature engineering transformer for LoanWise AI."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COLUMN = "Loan_Status"
TARGET_MAPPING = {"Y": 1, "N": 0}
ID_COLUMN = "Loan_ID"

RAW_FEATURE_COLUMNS: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

STRING_COLUMNS: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERIC_RAW_COLUMNS: List[str] = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]


def clean_string_series(series: pd.Series) -> pd.Series:
    """Strip surrounding whitespace from string values and normalize missing values to np.nan."""
    is_missing = series.isna()
    stripped = series.astype(str).str.strip()
    cleaned = stripped.where(~is_missing, np.nan)
    return cleaned.replace({"": np.nan, None: np.nan})


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer that cleans raw loan inputs and engineers TotalIncome.

    Stateless transformations ensure zero data leakage between train and test folds
    and guarantee identical behavior during single-row API inference.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None) -> None:
        self.feature_columns = feature_columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LoanFeatureEngineer":
        """Fit is a no-op because all cleaning and engineering steps are stateless."""
        self.expected_columns_ = list(self.feature_columns or RAW_FEATURE_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate schema, clean strings/Dependents, and compute TotalIncome."""
        expected_cols = getattr(self, "expected_columns_", list(self.feature_columns or RAW_FEATURE_COLUMNS))
        df = pd.DataFrame(X).copy()

        if ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])

        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required input column(s): {missing_cols}")

        df = df[expected_cols].copy()

        for col in STRING_COLUMNS:
            if col in df.columns:
                df[col] = clean_string_series(df[col])

        if "Dependents" in df.columns:
            dependents_clean = df["Dependents"].replace({"3+": "3"})
            df["Dependents"] = pd.to_numeric(dependents_clean, errors="coerce")

        for col in NUMERIC_RAW_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

        self.output_columns_ = df.columns.tolist()
        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return output column names for scikit-learn pipeline introspection."""
        cols = getattr(
            self,
            "output_columns_",
            list(self.feature_columns or RAW_FEATURE_COLUMNS) + ["TotalIncome"],
        )
        return np.asarray(cols, dtype=object)
