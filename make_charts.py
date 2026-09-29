"""Plot SHAP feature impact from outputs/feature_impact.csv. Run after train.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(output_dir: Path = Path("outputs"), docs_dir: Path = Path("docs")) -> None:
    impact = pd.read_csv(output_dir / "feature_impact.csv").head(6)[::-1]
    metrics = json.loads((output_dir / "metrics.json").read_text())
    docs_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(impact.feature, impact.mean_absolute_shap, color="#2f7fd8")
    for y, value in enumerate(impact.mean_absolute_shap):
        ax.text(value + 0.01, y, f"{value:.3f}", va="center")
    ax.set_title("SHAP feature impact (XGBoost)", loc="left", fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.text(
        0.01, 0.01,
        f"Holdout accuracy {metrics['test_accuracy']:.1%} | ROC-AUC {metrics['test_roc_auc']:.4f} | n={metrics['rows']:,}",
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(docs_dir / "shap_feature_impact.png", dpi=120)
    print(f"Wrote {docs_dir / 'shap_feature_impact.png'}")


if __name__ == "__main__":
    main()
