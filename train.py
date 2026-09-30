"""Train and evaluate an XGBoost churn model on the IBM Telco Customer Churn data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare import DATA_PATH, features, load

PARAM_GRID = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1, 5, 10],
}


def scores(y_true, probabilities) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    predictions = (probabilities >= 0.5).astype(int)
    ranked = np.asarray(y_true)[np.argsort(-probabilities)]
    top = ranked[: int(len(ranked) * 0.2)]
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "recall": round(float(recall_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions)), 4),
        "top20_churners_captured": round(float(top.sum() / ranked.sum()), 4),
        "top20_lift": round(float(top.mean() / ranked.mean()), 2),
    }


def train(data_path: Path, output_dir: Path, seed: int = 42) -> dict:
    import numpy as np
    import pandas as pd
    import shap
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
    from xgboost import XGBClassifier

    x, y = features(load(data_path))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=seed)

    # Tuning uses 5-fold CV on the training split only; the test split is scored once at the end.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=4),
        PARAM_GRID, n_iter=40, scoring="accuracy", cv=folds, random_state=seed, n_jobs=1,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_
    probabilities = model.predict_proba(x_test)[:, 1]

    # Same settings with SMOTE inside the pipeline, to show the recall / precision trade-off.
    smote = Pipeline([("smote", SMOTE(random_state=seed)),
                      ("model", XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=4, **search.best_params_))])
    smote.fit(x_train, y_train)

    metrics = {
        "dataset": "IBM Telco Customer Churn (7,043 customers)",
        "rows": int(len(x)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_churn_rate": round(float(y_test.mean()), 4),
        "majority_class_accuracy": round(float(1 - y_test.mean()), 4),
        "cv_accuracy": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
        "test": scores(y_test, probabilities),
        "test_with_smote": scores(y_test, smote.predict_proba(x_test)[:, 1]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"churned": y_test.to_numpy(), "score": probabilities.round(6)}).to_csv(
        output_dir / "holdout_scores.csv", index=False)
    sample = x_test.sample(n=min(500, len(x_test)), random_state=seed)
    impact = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(sample))).mean(axis=0)
    pd.DataFrame({"feature": x.columns, "mean_absolute_shap": impact}).sort_values(
        "mean_absolute_shap", ascending=False).to_csv(output_dir / "feature_impact.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output_dir, args.seed), indent=2))
