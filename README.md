# 🏦 Bank Loan Default Prediction — Decision Tree

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview
A bank wants to automate loan approval decisions. This project builds an interpretable Decision Tree classifier to predict which applicants are likely to default, balancing credit risk against approving creditworthy customers.

## Dataset
- Give Me Some Credit (Loan Default) — [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data)

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn (Decision Tree) · SQL-style analysis

## Approach
1. Explored loan applicant features and checked the (heavily imbalanced) default rate
2. Capped extreme outliers at the 99th percentile and imputed missing income/dependents data
3. Tuned tree depth (1–15) using cross-validation to find the bias-variance sweet spot
4. Trained the final Decision Tree with `class_weight='balanced'`
5. Visualized the tree structure for interpretability
6. Evaluated with precision, recall and AUC-ROC, prioritizing recall on defaulters

## Key Results & Insights
- Revolving credit-card utilization is the top predictor of default
- Prior 30–59 day delinquencies are the second strongest predictor — past behavior predicts future risk
- The default rate is only ~6.7%, requiring `class_weight='balanced'` to avoid a majority-class-biased model
- Optimal tree depth is 5–7; deeper trees clearly overfit on the learning curve
- Customers under 30 show a higher default rate than older age groups

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook loan_default_decision_tree.ipynb
```

## Project Structure
```
bank-loan-default-prediction-decision-tree/
├── notebooks/loan_default_decision_tree.ipynb
├── images/  (depth_tuning.png, decision_tree.png, feature_importance.png)
├── requirements.txt
└── README.md
```

## Author
Akshat Kesharwani — [LinkedIn](https://linkedin.com/in/akshat-kesharwani-b0452b346) · [Portfolio](https://akshatkesharwani-info.github.io)
