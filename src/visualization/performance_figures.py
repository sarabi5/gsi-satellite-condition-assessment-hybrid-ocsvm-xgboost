"""
Generate the overall and per-class model performance summary figures
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
OUTPUT_DIR = "outputs/figures"

COMPONENTS = [
    "Overall GSI\nPerformance",
    "Vegetation Health\nin Basin Area",
    "Vegetation\nStability",
    "Erosion within\nBasin Area",
    "Debris or Trash\nwithin Drainage Area",
]

OVERALL_METRICS = {
    "Accuracy": [0.61, 0.72, 0.93, 0.70, 0.78],
    "F1-score": [0.59, 0.77, 0.93, 0.73, 0.77],
    "AUC-ROC": [0.57, 0.65, 0.66, 0.59, 0.62],
    "Brier Score": [0.27, 0.19, 0.06, 0.21, 0.18],
}

PER_CLASS_METRICS = {
    "Overall GSI\nPerformance": {"Compliant": [0.50, 0.35, 0.41], "Non-Compliant": [0.65, 0.77, 0.70]},
    "Vegetation Health\nin Basin Area": {"Compliant": [0.90, 0.78, 0.84], "Non-Compliant": [0.32, 0.54, 0.40]},
    "Vegetation\nStability": {"Compliant": [0.96, 0.97, 0.97], "Non-Compliant": [0.22, 0.17, 0.19]},
    "Erosion within\nBasin Area": {"Compliant": [0.87, 0.75, 0.91], "Non-Compliant": [0.26, 0.43, 0.32]},
    "Debris or Trash\nwithin Drainage Area": {"Compliant": [0.86, 0.87, 0.86], "Non-Compliant": [0.36, 0.34, 0.35]},
}
# Each per-class list is [Precision, Recall, F1-score]


def _add_bar_labels(ax, bars, fontsize=8):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=fontsize)


def plot_overall_metrics(output_dir=OUTPUT_DIR):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    os.makedirs(output_dir, exist_ok=True)

    colors = ["#000000", "#555555", "#999999", "#cccccc"]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(COMPONENTS))
    width = 0.18

    for i, (metric, vals) in enumerate(OVERALL_METRICS.items()):
        bars = ax.bar(x + i * width, vals, width, label=metric, color=colors[i], edgecolor="white")
        _add_bar_labels(ax, bars)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(COMPONENTS, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance by Inspection Component")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "overall_metrics.png")
    plt.savefig(output_path, dpi=350)
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_class_metrics(output_dir=OUTPUT_DIR):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    os.makedirs(output_dir, exist_ok=True)

    class_colors = {"Compliant": "#444444", "Non-Compliant": "#aaaaaa"}
    class_metric_labels = ["Precision", "Recall", "F1-score"]

    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(18, 6), sharey=True)
    fig.subplots_adjust(top=0.78)

    for ax, (component, classes) in zip(axes, PER_CLASS_METRICS.items()):
        x = np.arange(len(class_metric_labels))
        for j, (cls, vals) in enumerate(classes.items()):
            bars = ax.bar(x + j * 0.35, vals, 0.35, label=cls, color=class_colors[cls], edgecolor="white")
            _add_bar_labels(ax, bars)
        ax.set_title(component, fontsize=9, pad=6)
        ax.set_xticks(x + 0.175)
        ax.set_xticklabels(class_metric_labels, fontsize=9)
        ax.set_ylim(0, 1.25)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)

    axes[0].set_ylabel("Score")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in class_colors.values()]
    fig.legend(handles, class_colors.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=2, fontsize=11, frameon=True)
    fig.suptitle("Per-Class Performance by Inspection Component", fontsize=12, y=1.02)

    output_path = os.path.join(output_dir, "per_class_metrics.png")
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_overall_metrics()
    plot_per_class_metrics()
