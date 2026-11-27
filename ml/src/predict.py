"""Model loading and single-application prediction helpers for LoanWise AI."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.src.feature_engineering import RAW_FEATURE_COLUMNS
from ml.src.train import (
    METADATA_FILENAME,
    MODEL_PIPELINE_FILENAME,
    SAVED_MODELS_DIR,
)

DISCLAIMER_TEXT = (
    "Educational prototype estimate based on historical data. "
    "Not an official banking or credit decision."
)


def load_model_artifacts(
    models_dir: Path = SAVED_MODELS_DIR,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load the serialized Joblib prediction pipeline and model metadata JSON."""
    model_path = models_dir / MODEL_PIPELINE_FILENAME
    metadata_path = models_dir / METADATA_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(
            f"Serialized model pipeline not found at '{model_path}'. "
            "Run 'python -m ml.src.train' first."
        )

    model_pipeline: Pipeline = joblib.load(model_path)
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    return model_pipeline, metadata


def build_input_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert a single applicant dictionary into a 1-row DataFrame in exact column order."""
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in payload]
    if missing:
        raise ValueError(f"Missing required input feature(s): {missing}")

    ordered_row = {col: payload[col] for col in RAW_FEATURE_COLUMNS}
    return pd.DataFrame([ordered_row], columns=RAW_FEATURE_COLUMNS)


def classify_confidence_band(approval_probability: float) -> Tuple[str, bool]:
    """Return confidence band label and human-review flag for borderline scores."""
    if 0.40 <= approval_probability <= 0.60:
        return "Uncertain / Borderline", True
    if approval_probability >= 0.75 or approval_probability <= 0.25:
        return "High", False
    return "Moderate", False


def predict_loan_application(
    model_pipeline: Pipeline,
    payload: Dict[str, Any],
    model_version: str = "v1.0.0",
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Generate prediction, probabilities, and engineered features for one applicant."""
    input_df = build_input_dataframe(payload)
    engineered_df = model_pipeline.named_steps["feature_engineer"].transform(input_df)

    approval_prob = float(model_pipeline.predict_proba(input_df)[0, 1])
    rejection_prob = 1.0 - approval_prob
    predicted_label = 1 if approval_prob >= threshold else 0
    prediction = "Approved" if predicted_label == 1 else "Rejected"
    confidence_band, human_review = classify_confidence_band(approval_prob)

    return {
        "prediction": prediction,
        "predicted_label": predicted_label,
        "approval_probability": round(approval_prob, 4),
        "rejection_probability": round(rejection_prob, 4),
        "confidence_band": confidence_band,
        "human_review_recommended": human_review,
        "engineered_features": {
            "TotalIncome": float(engineered_df.iloc[0]["TotalIncome"]),
        },
        "model_version": model_version,
        "disclaimer": DISCLAIMER_TEXT,
    }
