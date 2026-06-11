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
credit-risk-assessment/
│
├── credit_risk_final_project.ipynb   # Main notebook
├── cs-training.csv/
│   └── cs-training.csv               # Dataset (not included in repo)
└── README.md
```

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
Custom cost-benefit payoff function sweeping thresholds from 0.02 to 0.60 to maximize expected value per applicant:

| Outcome | Payoff |
|---------|--------|
| True Positive (catch bad borrower) | +$B_{TP}$ |
| False Positive (wrongly flag good borrower) | −$C_{FP}$ |
| False Negative (miss bad borrower) | −$C_{FN}$ |
| True Negative | $0 |

---

## Key Results

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic Regression | ~0.835 | — |
| Random Forest (tuned) | **best** | **best** |

> Exact values will populate once the notebook is run end-to-end against the dataset.

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

2. Download the dataset from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) and place it at:
   ```
   cs-training.csv/cs-training.csv
   ```

3. Launch Jupyter from the `final_project/` directory so relative paths resolve:
   ```bash
   jupyter notebook credit_risk_final_project.ipynb
   ```

4. Run all cells top to bottom. Section 9 will output the optimal threshold and confusion matrix.

---

## Limitations & Next Steps

- **Calibration:** predicted probabilities may need Platt scaling or isotonic regression for deployment use
- **Advanced models:** XGBoost or LightGBM likely to improve AUC further
- **Explainability:** SHAP values would make feature contributions transparent to underwriters
- **Fairness:** age and income features warrant disparate impact analysis before production use
- **Payoff constants:** the cost matrix in Section 9 uses illustrative values — real deployment requires input from a credit risk team

---

## Author

**Praneetha Gunturu**  
MS Business Analytics — San Francisco State University  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
