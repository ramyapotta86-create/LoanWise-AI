"""Leak-free scikit-learn preprocessing pipelines for LoanWise AI."""

import inspect
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.src.feature_engineering import (
    ID_COLUMN,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_MAPPING,
    LoanFeatureEngineer,
    clean_string_series,
)

CONTINUOUS_NUMERIC_COLUMNS: List[str] = [
    "Dependents",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "TotalIncome",
]

BINARY_NUMERIC_COLUMNS: List[str] = [
    "Credit_History",
]

CATEGORICAL_COLUMNS: List[str] = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
]


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder compatible across scikit-learn 1.0–1.4+."""
    params = inspect.signature(OneHotEncoder.__init__).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate raw 11 input features and binary target from a labeled dataset.

    - Drops exact duplicate rows.
    - Drops rows only if the target (`Loan_Status`) itself is missing.
    - Never drops rows for missing input feature values (handled inside Pipeline).
    - Excludes `Loan_ID` from the feature matrix.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' not found in DataFrame.")

    cleaned_df = df.drop_duplicates().copy()
    cleaned_df[TARGET_COLUMN] = clean_string_series(cleaned_df[TARGET_COLUMN])
    cleaned_df = cleaned_df.dropna(subset=[TARGET_COLUMN])

    unexpected_labels = set(cleaned_df[TARGET_COLUMN].unique()) - set(TARGET_MAPPING.keys())
    if unexpected_labels:
        raise ValueError(
            f"Unexpected target values in '{TARGET_COLUMN}': {sorted(unexpected_labels)}. "
            f"Expected {sorted(TARGET_MAPPING.keys())}."
        )

    y = cleaned_df[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)

    drop_cols = [TARGET_COLUMN]
    if ID_COLUMN in cleaned_df.columns:
        drop_cols.append(ID_COLUMN)

    X = cleaned_df.drop(columns=drop_cols)
    missing_features = [col for col in RAW_FEATURE_COLUMNS if col not in X.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature column(s): {missing_features}")

    X = X[RAW_FEATURE_COLUMNS].copy()
    return X, y


def build_column_transformer(
    scale_numeric: bool = True,
    categorical_columns: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build the ColumnTransformer for numerical and categorical preprocessing."""
    continuous_steps = [("imputer", SimpleImputer(strategy="median"))]
    binary_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]

    if scale_numeric:
        continuous_steps.append(("scaler", StandardScaler()))
        binary_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", _make_one_hot_encoder()),
    ]
    cat_cols = categorical_columns if categorical_columns is not None else CATEGORICAL_COLUMNS

    return ColumnTransformer(
        transformers=[
            ("num_continuous", Pipeline(continuous_steps), CONTINUOUS_NUMERIC_COLUMNS),
            ("num_binary", Pipeline(binary_steps), BINARY_NUMERIC_COLUMNS),
            ("categorical", Pipeline(categorical_steps), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessing_pipeline(
    scale_numeric: bool = True,
    feature_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
) -> Pipeline:
    """Build the full leak-free preprocessing pipeline (feature engineering + ColumnTransformer)."""
    return Pipeline(
        steps=[
            ("feature_engineer", LoanFeatureEngineer(feature_columns=feature_columns)),
            (
                "column_transformer",
                build_column_transformer(
                    scale_numeric=scale_numeric,
                    categorical_columns=categorical_columns,
                ),
            ),
        ]
    )


def get_transformed_feature_names(preprocessing_pipeline: Pipeline) -> List[str]:
    """Extract clean output feature names from a fitted preprocessing pipeline."""
    col_transformer = preprocessing_pipeline.named_steps["column_transformer"]
    return list(col_transformer.get_feature_names_out())


def map_transformed_name_to_raw(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to its original feature name."""
    if "__" in transformed_name:
        prefix, remainder = transformed_name.split("__", 1)
    else:
        prefix, remainder = "", transformed_name

    if prefix == "categorical":
        for cat_col in sorted(CATEGORICAL_COLUMNS, key=len, reverse=True):
            if remainder.startswith(f"{cat_col}_"):
                return cat_col
    return remainder
