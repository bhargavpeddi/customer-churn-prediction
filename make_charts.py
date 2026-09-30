"""Charts from the train.py outputs: SHAP drivers, cumulative gains, confusion matrix."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "tenure": "Tenure (months)",
    "Contract_Two year": "Two-year contract",
    "Contract_One year": "One-year contract",
    "InternetService_Fiber optic": "Fiber optic internet",
    "PaymentMethod_Electronic check": "Pays by electronic check",
    "PaperlessBilling_Yes": "Paperless billing",
    "TotalCharges": "Total charges",
    "MonthlyCharges": "Monthly charges",
    "avg_monthly_spend": "Average monthly spend",
    "add_on_services": "Number of add-on services",
}


def main(output_dir: Path = Path("outputs"), docs_dir: Path = Path("docs")) -> None:
    impact = pd.read_csv(output_dir / "feature_impact.csv").head(8)[::-1]
    test = json.loads((output_dir / "metrics.json").read_text())["test"]
    docs_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    style(ax)
    names = [LABELS.get(f, f.replace("_", ": ")) for f in impact.feature]
    ax.barh(names, impact.mean_absolute_shap, color=BLUE, height=0.6)
    for y, value in enumerate(impact.mean_absolute_shap):
        ax.text(value + 0.008, y, f"{value:.2f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, impact.mean_absolute_shap.max() * 1.15)
    ax.set_title("What drives churn: mean |SHAP value| on the test set", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("Mean |SHAP value| (log-odds)", color=INK)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="x", alpha=0.6, color=GRID)
    fig.text(0.01, 0.01,
             f"IBM Telco data, 1,409-customer test set: accuracy {test['accuracy']:.1%} | ROC-AUC {test['roc_auc']:.2f} | "
             f"top 20% of scores hold {test['top20_churners_captured']:.0%} of churners",
             color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(docs_dir / "shap_feature_impact.png", dpi=200)
    print(f"Wrote {docs_dir / 'shap_feature_impact.png'}")


INK, MUTED, GRID, SURFACE = "#1f2328", "#6b7280", "#e5e7eb", "#fcfcfb"
BLUE, BLUE_RAMP = "#2a78d6", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)


def gains_chart(output_dir: Path, docs_dir: Path) -> None:
    """Cumulative gains: share of churners reached when contacting the top X% of risk scores."""
    scores = pd.read_csv(output_dir / "holdout_scores.csv").sort_values("score", ascending=False)
    reached = scores["churned"].cumsum().to_numpy() / scores["churned"].sum()
    contacted = (pd.RangeIndex(1, len(scores) + 1) / len(scores)).to_numpy()
    top20 = reached[int(len(scores) * 0.2) - 1]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    style(ax)
    ax.plot([0, 100], [0, 100], linestyle="--", color=MUTED, linewidth=1.5, label="Random order")
    ax.plot(contacted * 100, reached * 100, color=BLUE, linewidth=2.5, label="Model risk score")
    ax.scatter([20], [top20 * 100], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"Top 20% of scores\nreach {top20:.0%} of churners", (20, top20 * 100), xytext=(3, top20 * 100 + 18),
                color=INK, fontsize=11, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Customers contacted, highest risk first (%)", color=INK)
    ax.set_ylabel("Churners reached (%)", color=INK)
    ax.set_title("Cumulative gains on the 1,409-customer holdout", loc="left", fontweight="bold", color=INK)
    ax.grid(alpha=0.6, color=GRID)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(docs_dir / "gains_curve.png", dpi=200)


def confusion_chart(output_dir: Path, docs_dir: Path) -> None:
    """Confusion matrix at the default 0.5 threshold."""
    scores = pd.read_csv(output_dir / "holdout_scores.csv")
    pred = scores["score"] >= 0.5
    actual = scores["churned"] == 1
    cells = [[int((~actual & ~pred).sum()), int((~actual & pred).sum())],
             [int((actual & ~pred).sum()), int((actual & pred).sum())]]
    total = sum(map(sum, cells))

    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    peak = max(map(max, cells))
    for r in range(2):
        for c in range(2):
            v = cells[r][c]
            shade = BLUE_RAMP[min(3, int(v / peak * 3.999))]
            ax.add_patch(plt.Rectangle((c + 0.02, 1 - r + 0.02), 0.96, 0.96, color=shade))
            ink = "white" if shade in ("#3987e5", "#1c5cab") else INK
            ax.text(c + 0.5, 1.5 - r + 0.07, f"{v:,}", ha="center", va="center", fontsize=22, fontweight="bold", color=ink)
            ax.text(c + 0.5, 1.5 - r - 0.15, f"{v / total:.1%} of customers", ha="center", va="center", fontsize=10, color=ink)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["Predicted: stays", "Predicted: churns"], color=INK)
    ax.set_yticks([1.5, 0.5], ["Actually stayed", "Actually churned"], color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Holdout confusion matrix (threshold 0.5)", loc="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(docs_dir / "confusion_matrix.png", dpi=200)


if __name__ == "__main__":
    main()
    gains_chart(Path("outputs"), Path("docs"))
    confusion_chart(Path("outputs"), Path("docs"))
    print("Wrote docs/gains_curve.png and docs/confusion_matrix.png")
