# Credit Card Default Prediction

## Project Overview

This project aims to build a predictive model that classifies whether a credit card customer is likely to default in the upcoming month. Early identification of such customers helps financial institutions proactively manage credit risk and mitigate potential losses.

The model was trained on historical customer data and deployed to generate predictions on unseen data for production use.

---

## Files

| File                  | Description                                         |
|-----------------------|-----------------------------------------------------|
| `train_dataset_final1.csv`   | Training dataset containing historical labeled records |
| `validate_dataset_final.csv`| Unlabeled validation dataset for final predictions |
| `EDA_and_model_workflow.ipynb` | Notebook covering EDA, feature engineering, and modeling |
| `default_predictions.csv`     | Final predictions generated for `validate_dataset.csv` |

---

## Problem Statement

The task is to predict whether a customer will default on their credit card in the next month using demographic, behavioral, and transactional data. The target variable is `next_month_default`, where:

- `0` = No Default  
- `1` = Default

The business challenge is rooted in **high class imbalance** — only **19.03%** of records represent default cases.

---

## Exploratory Data Analysis (EDA)

Key insights from data exploration:

- **Gender Distribution**: More male customers (sex = 1) than female (sex = 0).
- **Marital Status**: Majority of customers are married (marriage = 1 or 2).
- **Education Levels**: Predominantly education levels 1, 2, and 3; levels 0, 4–6 may represent data entry noise.
- **Age Distribution**: Right-skewed, most customers between 25–40 years.
- **Credit Limit (LIMIT_BAL)**: Highly right-skewed; a small number of customers have very high limits.
- **Credit Limit by Education**: No clear upward trend in limit with higher education; some inconsistency likely due to data noise.
- **Payment Status**: Majority have payment status 0 (on-time). Delays (>0) show a steep drop, but late payments correlate with defaults.
- **Average Bill vs. Payment Amount**: Most customers pay less than their average bill amount.

Plots used:
- Count plots (marriage, sex, education)
- Histograms and KDEs (age, credit limit)
- Boxplots (LIMIT_BAL vs education)
- Payment delay status histogram
- Overlayed density plots (bill vs payment)

---

## Feature Engineering

To enhance model signal, 18 custom features were engineered:

| Feature                     | Description |
|-----------------------------|-------------|
| `avg_payment`               | Mean of all payment amounts |
| `credit_utilization`        | `AVG_Bill_amt` divided by `LIMIT_BAL` |
| `delinquency_streak`        | Longest streak of consecutive delayed payments |
| `avg_payment_delay`         | Average payment delay (only positive values) |
| `payment_consistency`       | Fraction of months with non-zero payments |
| `has_overpaid`              | Flag for any month with negative bill (overpaid) |
| `missed_payments`           | Count of months with positive payment delay |
| `utilization_volatility`    | Std. dev. of bill amounts |
| `repayment_to_utilization` | Ratio of total payment to total bill amount |
| `full_payments`             | Count of months where payment ≥ bill |
| `payment_delay_std`         | Std. dev. of delay values |
| `bill_trend`                | Slope of bill amounts across months (trend) |
| `bill_skewness`             | Skewness of bill amount distribution |
| `max_payment_delay`         | Maximum recorded delay |
| `underpayments`             | Count of months with payment < bill |
| `zero_spending_months`      | Number of months with zero bill amount |
| `late_payment_ratio`        | Fraction of months with delay (`pay_X > 0`) |
| `no_usage_flag`             | Flag where total bill amount over 6 months is zero |

These features captured behavior patterns such as payment discipline, over-utilization, delayed habits, and no-activity accounts.

---

## Modeling and Evaluation

Multiple models were trained using resampled data and calibrated probabilities. Class imbalance was tackled using:

- SMOTE oversampling
- Class weights
- Custom threshold tuning
- F2 score optimization (recall-focused)

The final classification was based on a **custom threshold (e.g. 0.45)** instead of default 0.5 to maximize F2.

### Model Comparison Table

| Model               | Recall on Class 1 | F2 Score | Best Threshold |
|---------------------|------------------|----------|----------------|
| Logistic Regression | 80%            | 0.5970  | 0.36           |
| Random Forest       | 88%            | 0.6120  | 0.19           |
| XGBoost             | 80%            | 0.5859  | 0.23           |
| LightGBM            | 77%            | 0.5880  | 0.30           |
| **CatBoost**        | **86%**        | **0.6009** | **0.45** (Final Model) |

CatBoost was selected as the final model due to its superior performance, realistic best threshold and minimal preprocessing requirements.

---

## Final Predictions

The model was applied to `validate_dataset_final.csv` to generate final predictions. The output file, `default_predictions.csv`, includes:

- `Customer_ID`
- `next_month_default` (predicted class using threshold = 0.45)

This output is ready for deployment or evaluation by business stakeholders.

---

## Technologies Used

- Python (pandas, NumPy, matplotlib, seaborn)
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost, LightGBM, CatBoost
- Jupyter Notebook

---

## Summary

This project demonstrates a complete credit risk modeling pipeline — from raw data to interpretable features, rigorous model evaluation, and production-ready predictions. Emphasis was placed on **domain-aware feature engineering**, **recall-centric metrics**, and **calibrated classification** to maximize business value.

