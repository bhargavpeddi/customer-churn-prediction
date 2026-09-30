"""Train and evaluate a leakage-safe churn model on synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_data import FEATURES, write_csv


def train(data_path: Path, output_dir: Path, seed: int = 42) -> dict[str, float | int | str]:
    # Imported here so data-generation tests run without ML dependencies installed.
    import numpy as np
    import pandas as pd
    import shap
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from xgboost import XGBClassifier

    frame = pd.read_csv(data_path)
    required = {"customer_id", *FEATURES, "churned"}
    if set(frame.columns) != required or frame.isna().any().any():
        raise ValueError("Input schema is invalid or contains missing values")
    if frame.customer_id.duplicated().any() or not set(frame.churned.unique()).issubset({0, 1}):
        raise ValueError("Customer IDs must be unique and target must be binary")

    features = frame[list(FEATURES)]
    target = frame["churned"].astype(int)
    if target.value_counts().min() < 12:
        raise ValueError("Each class needs at least 12 observations")
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=seed
    )
    # SMOTE lives inside the CV pipeline: synthetic neighbors never reach validation/test folds.
    pipeline = Pipeline(
        [
            ("smote", SMOTE(random_state=seed)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=2,
                    random_state=seed,
                ),
            ),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "model__n_estimators": [100, 180],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.08],
            "model__subsample": [0.8, 1.0],
        },
        n_iter=6,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
        random_state=seed,
    )
    search.fit(x_train, y_train)
    probabilities = search.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, float | int | str] = {
        "dataset": "synthetic telecom demo; not the original project data",
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "cv_roc_auc": round(float(search.best_score_), 4),
        "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "test_recall": round(float(recall_score(y_test, predictions)), 4),
        "test_precision": round(float(precision_score(y_test, predictions)), 4),
        "baseline_churn_rate": round(float(y_test.mean()), 4),
    }
    # Retention teams work a ranked list, not a 0/1 label: how many of the actual
    # churners fall in the top 20% of scores, and how that compares to random.
    ranked = y_test.to_numpy()[np.argsort(-probabilities)]
    top = ranked[: int(len(ranked) * 0.2)]
    metrics["top20_churners_captured"] = round(float(top.sum() / ranked.sum()), 4)
    metrics["top20_lift"] = round(float(top.mean() / ranked.mean()), 2)
    # Per-customer holdout scores, used by make_charts.py for the gains curve and confusion matrix.
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"churned": y_test.to_numpy(), "score": probabilities.round(6)}).to_csv(
        output_dir / "holdout_scores.csv", index=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = x_test.sample(n=min(200, len(x_test)), random_state=seed)
    explainer = shap.TreeExplainer(search.best_estimator_.named_steps["model"])
    shap_values = np.asarray(explainer.shap_values(sample))
    mean_impact = np.abs(shap_values).mean(axis=0)
    pd.DataFrame({"feature": list(FEATURES), "mean_absolute_shap": mean_impact}).sort_values(
        "mean_absolute_shap", ascending=False
    ).to_csv(output_dir / "feature_impact.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/telecom_synthetic.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=7043)
    args = parser.parse_args()
    if not args.data.exists():
        write_csv(args.data, args.rows, args.seed)
    print(json.dumps(train(args.data, args.output_dir, args.seed), indent=2))
