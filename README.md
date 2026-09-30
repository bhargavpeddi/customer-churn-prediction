# Customer Churn Prediction

XGBoost model that predicts which telecom customers will cancel, with SMOTE for class imbalance and SHAP to explain the predictions.

📝 Write-up on Medium: [Predicting Customer Churn Without Fooling Yourself](https://medium.com/@bhargavpeddi/predicting-customer-churn-without-fooling-yourself-smote-xgboost-and-shap-0f65369e73d6)

![SHAP feature impact](docs/shap_feature_impact.png)

## Results

Seed 42, 7,043 customers, stratified 80/20 split, 16.1% of customers churn.

| Metric | Value |
| --- | --- |
| Churners caught (recall) | 61.7% |
| Churners in the top 20% of risk scores | 47.6% (2.4x lift over random) |
| Holdout ROC-AUC | 0.7497 |
| Cross-validation ROC-AUC (train) | 0.7523 |
| Precision at 0.5 threshold | 31.9% |
| Accuracy | 72.6% |

Recall and top-20% capture are the numbers that matter for retention: a team calling the 20% highest-risk customers reaches almost half of everyone who would churn. Accuracy is a poor metric here, because predicting "nobody churns" already scores 83.9%; SMOTE deliberately trades some accuracy for catching more churners. CV and holdout ROC-AUC are within 0.003 of each other, so the model is not overfitting.

Monthly charges, autopay, and tenure have the largest SHAP impact.

| Cumulative gains | Confusion matrix (threshold 0.5) |
| --- | --- |
| ![Cumulative gains](docs/gains_curve.png) | ![Confusion matrix](docs/confusion_matrix.png) |

## Data

`generate_data.py` builds a seeded dataset with a standard telecom schema: tenure, monthly charges, support tickets, late payments, contract length, fiber, autopay, and streaming services. I used generated data because the customer data I've worked with can't be published. The same seed always produces the same rows, so the numbers above are reproducible.

## How it works

1. Validate the schema: required columns, no missing values, unique customer IDs, binary target.
2. Split off a stratified 20% holdout before any fitting.
3. Tune XGBoost with `RandomizedSearchCV` on the training set. SMOTE sits inside the pipeline, so oversampled rows never reach a validation fold.
4. Score the holdout once, then compute mean absolute SHAP values on a holdout sample.

## Run it

Python 3.9+. On macOS, XGBoost needs OpenMP: `brew install libomp`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py            # writes outputs/metrics.json (incl. recall, lift) and outputs/feature_impact.csv
python make_charts.py      # writes the charts in docs/ (needs matplotlib)
python -m unittest -v
```

`python train.py --help` lists options for the data path, row count, seed, and output folder.

## Next steps

- Time-based validation instead of a random split
- Probability calibration so scores can drive retention budgets
- Drift monitoring on the input features
