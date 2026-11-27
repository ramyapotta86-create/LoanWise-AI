# Saved Model Artifacts

Serialized `.joblib` files are excluded from Git tracking via `.gitignore`.

After downloading the primary Kaggle dataset into `data/raw/primary/`, run from the repository root (`LoanWise-AI/`):

```bash
python -m ml.src.train
```

This generates:
- `model_pipeline.joblib` — Full preprocessing + winning classifier pipeline
- `preprocessor.joblib` — Standalone fitted preprocessing pipeline
- `shap_background.joblib` — Compact transformed training sample for SHAP
- `model_metadata.json` — Measured CV/test evaluation metrics and schema metadata
