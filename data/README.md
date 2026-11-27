# Dataset Instructions — LoanWise AI

## Primary Assigned Dataset

- **Dataset:** Loan Prediction Problem Dataset (`altruistdelhite04`)
- **Kaggle URL:** https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
- **Target Column:** `Loan_Status` (`Y` = Approved, `N` = Rejected)

## How to Download Locally

1. Open the Kaggle URL above in your browser and download the dataset ZIP file.
2. Extract the CSV file(s) into:
   `data/raw/primary/`
3. Keep the CSV files local only — `data/raw/**/*.csv` is excluded by `.gitignore` so large/external datasets are not committed to GitHub.

## Inspect the Actual Downloaded Files

After activating your virtual environment, run:

```bash
python -m ml.src.data_loader
```

This script discovers the CSV file(s) inside `data/raw/primary/` and prints the real row count, columns, data types, missing values, duplicates, and target class distribution.
