# Customer Churn Prediction — reproducible demo

This project demonstrates the workflow behind a telecom churn project on my resume: feature engineering, class-imbalance handling, XGBoost tuning, held-out evaluation, and SHAP explanations. The generated sample has 7,043 **synthetic** customers. It is not the original telecom dataset, and its results must not be confused with metrics from the resume.

## Run it

Requires Python 3.9+. The workflow was run successfully with Python 3.9 and 3.12 on macOS.

On macOS, XGBoost also needs the OpenMP runtime: `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
python -m unittest -v
```

`train.py` writes `outputs/metrics.json` and `outputs/feature_impact.csv`. Override data, sample size, or output location with `python train.py --help`.

## Method and checks

The dataset generator produces tenure, charges, service, payment, and support features with a seeded churn outcome. Customer IDs are excluded from features. The split is stratified and fixed before fitting. SMOTE is inside the cross-validation pipeline, avoiding resampling leakage into validation folds. Hyperparameters are selected on training data only. Accuracy and ROC-AUC are then computed once on the untouched holdout; SHAP ranks features on a small holdout sample.

For a real deployment, I would add temporal validation, calibration, drift monitoring, fairness review, and a documented intervention policy. This repository contains no customer information or employer-owned code.
