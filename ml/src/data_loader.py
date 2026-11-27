"""Dataset discovery, loading, and initial schema inspection utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "primary"
TARGET_COLUMN = "Loan_Status"


def list_primary_csvs(data_dir: Path = PRIMARY_RAW_DIR) -> List[Path]:
    """Return sorted CSV file paths found inside the primary raw data folder."""
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))


def find_training_csv(
    data_dir: Path = PRIMARY_RAW_DIR,
    target_column: str = TARGET_COLUMN,
) -> Path:
    """Locate the labeled training CSV containing the target column.

    Checks CSV headers dynamically instead of hardcoding a filename.
    """
    csv_files = list_primary_csvs(data_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Please download the Kaggle dataset "
            "(https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) "
            "and place the extracted CSV file(s) into data/raw/primary/."
        )

    for csv_path in csv_files:
        header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if target_column in header_cols:
            return csv_path

    found_names = ", ".join(p.name for p in csv_files)
    raise ValueError(
        f"Found CSV file(s) [{found_names}] in '{data_dir}', "
        f"but none contain the expected target column '{target_column}'."
    )


def find_unlabeled_test_csv(
    data_dir: Path = PRIMARY_RAW_DIR,
    target_column: str = TARGET_COLUMN,
) -> Optional[Path]:
    """Locate an unlabeled Kaggle test CSV (without target_column), if present."""
    for csv_path in list_primary_csvs(data_dir):
        header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if target_column not in header_cols:
            return csv_path
    return None


def load_training_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the labeled training CSV into a pandas DataFrame."""
    path = csv_path if csv_path is not None else find_training_csv()
    return pd.read_csv(path)


def summarize_dataset(df: pd.DataFrame, source_name: str = "") -> Dict[str, Any]:
    """Collect and return basic factual summary statistics from a DataFrame."""
    missing_counts = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)

    summary: Dict[str, Any] = {
        "source_name": source_name,
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_counts": {col: int(cnt) for col, cnt in missing_counts.items()},
        "missing_percent": {col: float(pct) for col, pct in missing_pct.items()},
    }

    if "Loan_ID" in df.columns:
        summary["unique_loan_ids"] = int(df["Loan_ID"].nunique(dropna=False))

    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts(dropna=False)
        proportions = (df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100).round(2)
        summary["target_counts"] = {str(k): int(v) for k, v in counts.items()}
        summary["target_percent"] = {str(k): float(v) for k, v in proportions.items()}

    return summary


def print_dataset_report(df: pd.DataFrame, source_name: str = "") -> None:
    """Print a beginner-friendly inspection report for the loaded dataset."""
    info = summarize_dataset(df, source_name=source_name)
    print("=" * 60)
    print(f"Dataset Report: {info['source_name'] or 'Loaded DataFrame'}")
    print("=" * 60)
    print(f"Rows: {info['num_rows']} | Columns: {info['num_columns']}")
    print(f"Exact duplicate rows: {info['duplicate_rows']}")
    if "unique_loan_ids" in info:
        print(f"Unique Loan_ID values: {info['unique_loan_ids']}")

    print("\nColumns, Data Types, and Missing Values:")
    for col in info["columns"]:
        dtype = info["dtypes"][col]
        miss_c = info["missing_counts"][col]
        miss_p = info["missing_percent"][col]
        print(f"  - {col:20s} | dtype={dtype:10s} | missing={miss_c} ({miss_p}%)")

    if "target_counts" in info:
        print(f"\nTarget Column ({TARGET_COLUMN}) Distribution:")
        for label, count in info["target_counts"].items():
            pct = info["target_percent"][label]
            print(f"  - {label}: {count} ({pct}%)")

    print("\nFirst 5 rows:")
    print(df.head().to_string())


if __name__ == "__main__":
    try:
        train_path = find_training_csv()
        train_df = load_training_data(train_path)
        print_dataset_report(train_df, source_name=train_path.name)

        test_path = find_unlabeled_test_csv()
        if test_path is not None:
            test_df = pd.read_csv(test_path)
            print_dataset_report(test_df, source_name=test_path.name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Setup Needed] {exc}")
