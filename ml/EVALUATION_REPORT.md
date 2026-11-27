# LoanWise AI — ML Evaluation & Fairness Report

> **Status:** Verified on local Kaggle dataset (`train_u6lujuX_CVtuZ9i.csv`, `614` rows × `13` columns). All numbers below come directly from `python -m ml.src.data_loader`, `python -m ml.src.train`, and `python -m ml.src.fairness`.

---

## 1. Dataset Verification Summary

- **Source:** `altruistdelhite04/loan-prediction-problem-dataset`
- **Training CSV:** `train_u6lujuX_CVtuZ9i.csv` — `614` rows × `13` columns
- **Unlabeled Kaggle Test CSV:** `test_Y3wMUE5_7gLdaTN.csv` — `367` rows × `12` columns (no `Loan_Status` column, used only for inference sanity checks)
- **Exact duplicate rows:** `0` (`614` unique `Loan_ID` values)
- **Missing values in training set:**
  - `Credit_History`: `50` (`8.14%`)
  - `Self_Employed`: `32` (`5.21%`)
  - `LoanAmount`: `22` (`3.58%`)
  - `Dependents`: `15` (`2.44%`)
  - `Loan_Amount_Term`: `14` (`2.28%`)
  - `Gender`: `13` (`2.12%`)
  - `Married`: `3` (`0.49%`)
- **Verified category spellings:** `Gender` (`Male`, `Female`), `Married` (`Yes`, `No`), `Dependents` (`0`, `1`, `2`, `3+`), `Education` (`Graduate`, `Not Graduate`), `Self_Employed` (`No`, `Yes`), `Property_Area` (`Semiurban`, `Urban`, `Rural`), `Credit_History` (`1.0`, `0.0`)
- **Numerical skewness & IQR outlier counts (retained in pipeline with median imputation):**
  - `ApplicantIncome`: min `150`, median `3812.5`, max `81000`, skew `6.54`, IQR outliers `50`
  - `CoapplicantIncome`: min `0.0`, median `1188.5`, max `41667.0`, skew `7.49`, IQR outliers `18`
  - `LoanAmount`: min `9.0`, median `128.0`, max `700.0`, skew `2.68`, IQR outliers `39`
  - `Loan_Amount_Term`: min `12.0`, median `360.0`, max `480.0`, skew `-2.36`, IQR outliers `88`
- **Target (`Loan_Status`) class distribution:** `Y` (`1`, Approved) = `422` (`68.73%`), `N` (`0`, Rejected) = `192` (`31.27%`)

---

## 2. Model Comparison Results (`python -m ml.src.train`)

All 3 models were evaluated using an **80% Development (`491` rows) / 20% Held-Out Test (`123` rows)** stratified split (`random_state=42`) and **5-Fold Stratified Cross-Validation** inside the 80% development set:

| Model | Best CV Params | CV Accuracy (Mean ± SD) | CV F1 (Mean ± SD) | CV ROC-AUC (Mean ± SD) | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Overfitting Gap (F1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=0.1, class_weight=None` | `0.8003 ± 0.0458` | `0.8714 ± 0.0271` | `0.7268 ± 0.0904` | `0.8618` | `0.8400` | `0.9882` | `0.9081` | `0.8452` | `-0.0370` |
| Decision Tree | `max_depth=3, min_samples_leaf=15, class_weight=None` | `0.7901 ± 0.0610` | `0.8641 ± 0.0388` | `0.7008 ± 0.0750` | `0.8537` | `0.8317` | `0.9882` | `0.9032` | `0.7687` | `-0.0333` |
| Random Forest | `n_estimators=200, max_depth=4, min_samples_leaf=5, class_weight=None` | `0.7983 ± 0.0457` | `0.8703 ± 0.0271` | `0.7546 ± 0.0823` | `0.8537` | `0.8317` | `0.9882` | `0.9032` | `0.8381` | `-0.0325` |

- **Selected Winning Model:** `LogisticRegression` (`C=0.1`, `class_weight=None`) — highest 5-fold CV F1 (`0.8714`), highest held-out test Accuracy (`0.8618`), F1 (`0.9081`), and ROC-AUC (`0.8452`).
- **Held-Out Test Confusion Matrix (`123` rows):**
  - `[[TN=22, FP=16], [FN=1, TP=84]]`
  - **Rejected (`0`, support `38`):** Precision `0.9565`, Recall `0.5789`, F1 `0.7213`
  - **Approved (`1`, support `85`):** Precision `0.8400`, Recall `0.9882`, F1 `0.9081`
- **Key Error-Cost Observation:** Only `1` actual approval was rejected (`FN=1`), while `16` actual rejections were predicted as approved (`FP=16`, mostly applicants with `Credit_History=1` who were historically rejected for other reasons). This confirms why the borderline human-review flag (`0.40 <= p <= 0.60`) and manual underwriting disclaimer are important.

---

## 3. Global SHAP Feature Importance (`LinearExplainer` on `LogisticRegression`)

Top aggregated mean `|SHAP|` contributions across the 100-row transformed background sample:

1. `Credit_History`: `0.8183`
2. `Property_Area`: `0.2208`
3. `Married`: `0.1805`
4. `Education`: `0.1061`
5. `Dependents`: `0.0804`
6. `CoapplicantIncome`: `0.0522`
7. `Gender`: `0.0307`
8. `Self_Employed`: `0.0296`
9. `TotalIncome`: `0.0135`
10. `LoanAmount`: `0.0128`
11. `ApplicantIncome`: `0.0080`
12. `Loan_Amount_Term`: `0.0012`

---

## 4. Responsible AI & Demographic Fairness Ablation (`python -m ml.src.fairness`)

| Configuration | Test Accuracy | Test F1 | Test ROC-AUC | `Gender` DPD | `Gender` EOD | `Married` DPD | `Married` EOD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model A (All 11 Features) | `0.8618` | `0.9081` | `0.8452` | `0.1133` | `0.0143` | `0.1689` | `0.0400` |
| Model B (Without `Gender`, `Married`) | `0.8618` | `0.9081` | `0.8412` | `0.1133` | `0.0143` | `0.1689` | `0.0400` |

### Observed Test-Set Subgroup Breakdown (Model A)
- **`Gender`:** `Female` (`n=25`: actual approval `0.5600`, predicted `0.7200`, TPR `1.0000`); `Male` (`n=96`: actual approval `0.7292`, predicted `0.8333`, TPR `0.9857`); `<MISSING>` (`n=2`).
- **`Married`:** `No` (`n=44`: actual approval `0.5682`, predicted `0.7045`, TPR `0.9600`); `Yes` (`n=79`: actual approval `0.7595`, predicted `0.8734`, TPR `1.0000`).

### Documented Fairness & Scope Limitations
1. Predictions reflect patterns in a small historical public dataset (`614` rows) and **not** official banking creditworthiness.
2. Small test subgroup sizes (especially `Female`, `n=25`) make fairness metrics noisy and sensitive to individual records.
3. Removing `Gender` and `Married` (Model B) yielded the same test predictions and DPD/EOD here because correlated features (such as `Credit_History`, `CoapplicantIncome`, and `Property_Area`) still carry historical group differences.
4. SHAP explanations describe feature contributions to the **model's** score, **not** real-world banking causation.
5. Borderline probabilities (`0.40 <= p <= 0.60`) should trigger human review rather than automated decisioning.
