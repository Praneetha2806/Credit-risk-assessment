# Credit Risk Assessment Model
### DS 862 Final Project — Predicting Serious Loan Delinquency

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange) ![Status](https://img.shields.io/badge/status-complete-brightgreen)

---

## Overview

This project builds a binary classification model to predict whether a borrower will experience serious delinquency (90+ days past due) within two years — a core task in credit underwriting and lending limit decisions.

Using the Kaggle **Give Me Some Credit** dataset (`cs-training.csv`), the pipeline covers end-to-end model development: EDA, preprocessing, baseline and tuned models, threshold optimization via a cost-benefit framework, and evaluation via ROC and Precision-Recall curves.

---

## Business Problem

> **Target variable:** `SeriousDlqin2yrs` — did this borrower go 90+ days delinquent within 2 years?

Lenders use models like this to decide whether to approve a loan, set a credit limit, or flag an account for intervention. A missed bad borrower (false negative) is costly; an over-flagged good borrower (false positive) erodes revenue and customer trust. The optimal decision threshold is determined by balancing these asymmetric costs.

---

## Project Structure

```
final_project/
│
├── code/
│   └── credit_risk_final_project.ipynb   # Main notebook (run from inside this folder)
├── cs-training.csv                       # Dataset — place directly here, NOT in a subfolder
└── README.md
```

> The notebook resolves the dataset path as `DATA_PATH = Path("..") / "cs-training.csv"` — i.e. one level up from wherever the notebook is running. Place the CSV in `final_project/`, not inside `code/`.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) |
| Target | `SeriousDlqin2yrs` (binary: 0 / 1) |
| Features | 10 financial/demographic variables |
| Class imbalance | ~6.7% positive rate |

Key features include revolving utilization, age, number of open credit lines, past-due history (30/60/90 days), debt ratio, monthly income, and number of dependents.

---

## Methodology

### 1. EDA
- Missing value analysis (primary gaps in `MonthlyIncome` and `NumberOfDependents`)
- Class distribution visualization
- Correlation heatmap across all features

### 2. Preprocessing
- Drop row ID column
- Cap extreme values: revolving utilization clipped at 1.5, debt ratio at 10.0
- Invalid ages (≤ 0) set to NaN, then imputed
- Median imputation for all numeric features via `sklearn` pipeline
- Standard scaling before model fitting

### 3. Models
| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline; `class_weight='balanced'`, `max_iter=2000` |
| Random Forest (initial) | `class_weight='balanced_subsample'` |
| Random Forest (tuned) | `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf` |

### 4. Evaluation
- 5-fold Stratified Cross-Validation
- ROC-AUC and PR-AUC as primary metrics (preferred over accuracy given class imbalance)
- ROC and Precision-Recall curve plots for the primary model

### 5. Threshold Optimization
Custom cost-benefit payoff function sweeping thresholds from 0.02 to 0.60 to maximize expected value per applicant. Payoffs are business-justified dollar values rather than symbolic placeholders:

| Outcome | Payoff |
|---------|--------|
| True Positive (catch bad borrower) | +$500 |
| False Positive (wrongly flag good borrower) | −$50 |
| False Negative (miss bad borrower) | −$2,500 |
| True Negative | $0 |

This is roughly a 1 : 5 : 50 ratio — missing a default costs 5× the benefit of catching one, and 50× the cost of a false alarm.

---

## Key Results

| Model | ROC-AUC (test) | PR-AUC (test) |
|-------|---------|--------|
| Logistic Regression | 0.8332 | 0.3267 |
| Random Forest (initial, untuned) | 0.8355 | 0.3407 |
| Random Forest (tuned via GridSearchCV) | *pending* | *pending* |

**Confusion matrix @ 0.50 threshold (default cutoff):**

*Logistic Regression*
```
              precision    recall  f1-score   support
           0     0.9765    0.7627    0.8565     27995
           1     0.1834    0.7441    0.2943      2005

confusion matrix:
 [[21353  6642]
 [  513  1492]]
```

*Random Forest (initial)*
```
              precision    recall  f1-score   support
           0     0.9431    0.9902    0.9661     27995
           1     0.5477    0.1661    0.2549      2005

confusion matrix:
 [[27720   275]
 [ 1672   333]]
```

> The Random Forest is tuned via 5-fold `GridSearchCV` over `n_estimators`, `max_depth`, and `min_samples_leaf` (60 fits). The tuned model's ROC-AUC/PR-AUC, its ROC/Precision-Recall curves, and the cost-benefit threshold sweep (Section 9 of the notebook, using the $500 / −$50 / −$2,500 payoffs above) are still pending a completed end-to-end run — this section will be updated with the optimal threshold, the resulting confusion matrix, and the expected-cost reduction vs. the default 0.50 cutoff once that finishes.

Recommended threshold determined by expected-value sweep; confusion matrix and classification report provided at both the default 0.5 and the business-optimal threshold.

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Steps

1. Clone the repo and navigate into it:
   ```bash
   git clone https://github.com/your-username/credit-risk-assessment.git
   cd credit-risk-assessment
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) and place `cs-training.csv` directly inside the `final_project/` folder (one level up from the notebook) — **not** in a `data/` or `cs-training.csv/` subfolder:
   ```
   final_project/cs-training.csv
   ```

3. Launch Jupyter from inside the `code/` directory so the relative path `Path("..") / "cs-training.csv"` resolves correctly:
   ```bash
   cd final_project/code
   jupyter notebook credit_risk_final_project.ipynb
   ```

4. Run all cells top to bottom. Section 9 will output the optimal threshold and confusion matrix.

---

## Limitations & Next Steps

- **Calibration:** predicted probabilities may need Platt scaling or isotonic regression for deployment use
- **Advanced models:** XGBoost or LightGBM likely to improve AUC further
- **Explainability:** SHAP values would make feature contributions transparent to underwriters
- **Fairness:** age and income features warrant disparate impact analysis before production use
- **Payoff constants:** the cost matrix in Section 9 uses business-justified illustrative values (+$500 / −$50 / −$2,500) rather than real, audited loss data — actual deployment requires input from a credit risk team to calibrate these

---

## Author

**Praneetha Gunturu**  
MS Business Analytics — San Francisco State University  
[LinkedIn](https://www.linkedin.com/in/gunturu-praneetha/) 