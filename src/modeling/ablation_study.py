"""
Ablation study comparing XGBoost, SMOTE + XGBoost, and OCSVM + SMOTE + XGBoost using stratified 5-fold cross-validation.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score

from hybrid_model import OCSVM_NU_GRID, XGB_PARAM_GRID, OCSVMAnomalyFeature, make_xgb_classifier

warnings.filterwarnings("ignore")

# --- CONFIG ---
DATA_PATH = "data/quantile_ml_ready_dataset.csv"
RESULTS_DIR_ROOT = "outputs/ablation_study"

N_ITER_SEARCH = 60
N_SPLITS = 5
RANDOM_STATE = 42

# One entry per inspection target modeled in the manuscript. 'column' is the
# target's rating column name in the model-ready dataset produced by
# build_site_features.py.
TARGET_CONFIGS = [
    {"label": "Overall GSI Performance", "column": "OverallSMPFunctionalRatingId"},
    {"label": "Vegetation Health in Basin Area", "column": "VegetationRatingId(Ranking)"},
    {"label": "Vegetation Stability", "column": "SMPVegetativelyStabilizedId(Ranking)"},
    {"label": "Erosion within Basin Area", "column": "SignOfErosionId(Ranking)"},
    {"label": "Debris or Trash within Drainage Area", "column": "DebrisTrashWithinDrainageAreaId(Ranking)"},
]

PLOT_STYLE = {"font.family": "serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
C_DARK, C_MID, C_LIGHT, C_DASH = "#1a1a1a", "#555555", "#aaaaaa", "#888888"

VARIANT_ORDER = ["XGBoost Only", "SMOTE + XGBoost", "OCSVM + SMOTE + XGBoost (Hybrid)"]
VARIANT_COLORS = {"XGBoost Only": C_LIGHT, "SMOTE + XGBoost": C_MID, "OCSVM + SMOTE + XGBoost (Hybrid)": C_DARK}
HYBRID_NAME = "OCSVM + SMOTE + XGBoost (Hybrid)"


def build_variants(smote_strategy):
    hybrid_param_grid = {"ocsvm_feature__nu": OCSVM_NU_GRID, **XGB_PARAM_GRID}
    return {
        "XGBoost Only": dict(pipeline=Pipeline([("classifier", make_xgb_classifier())]), param_grid=XGB_PARAM_GRID),
        "SMOTE + XGBoost": dict(
            pipeline=Pipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=smote_strategy)),
                ("classifier", make_xgb_classifier()),
            ]),
            param_grid=XGB_PARAM_GRID,
        ),
        HYBRID_NAME: dict(
            pipeline=Pipeline([
                ("ocsvm_feature", OCSVMAnomalyFeature()),
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=smote_strategy)),
                ("classifier", make_xgb_classifier()),
            ]),
            param_grid=hybrid_param_grid,
        ),
    }


def prepare_features(raw_df, feature_columns):
    df = raw_df.copy()
    df["InspectionDate"] = pd.to_datetime(df["InspectionDate"])
    df["Asset_Age"] = df["InspectionDate"].dt.year - df["DATE_INSTALLED"]
    df.loc[df["Asset_Age"] < 0, "Asset_Age"] = pd.NA
    df = pd.get_dummies(df, columns=["GSI Type"], prefix="GSI", dtype=int)
    df = df.drop(columns=["DATE_INSTALLED", "InspectionDate"])

    gsi_cols = [c for c in df.columns if c.startswith("GSI_")]
    base_features = [c for c in feature_columns if c in df.columns] + ["Asset_Age"] + gsi_cols
    return df, base_features


def run_ablation_for_target(cfg, raw_df, base_features):
    label, target_col = cfg["label"], cfg["column"]
    target_slug = label.replace(" ", "_").replace("/", "-")
    results_dir = os.path.join(RESULTS_DIR_ROOT, target_slug)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'=' * 70}\nTarget: {label}  (column: {target_col})\n{'=' * 70}")

    df = raw_df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].map({1: 0, 2: 0, 3: 1, 4: 1})

    features = [c for c in base_features if c != target_col]
    X = df[features].copy().fillna(df[features].median())
    y = df[target_col]

    if y.nunique() < 2:
        print(f"  Skipped: target has fewer than 2 classes after mapping.")
        return None

    class_counts = y.value_counts()
    smote_target = int(class_counts.max())
    smote_strategy = {0: smote_target, 1: smote_target}
    print(f"  Class 0: {class_counts.get(0, 0)} | Class 1: {class_counts.get(1, 0)}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    variants = build_variants(smote_strategy)

    summary_rows, fold_rows = [], []
    for name, vcfg in variants.items():
        print(f"  Running variant: {name}")
        search = RandomizedSearchCV(
            estimator=vcfg["pipeline"],
            param_distributions=vcfg["param_grid"],
            n_iter=N_ITER_SEARCH,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            random_state=RANDOM_STATE,
        )
        search.fit(X, y)
        best_model = search.best_estimator_

        y_pred = cross_val_predict(best_model, X, y, cv=cv)
        y_prob = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")[:, 1]

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")
        auc = roc_auc_score(y, y_prob)
        brier = brier_score_loss(y, y_prob)

        prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(y, y_pred, labels=[0, 1], zero_division=0)

        summary_rows.append({
            "Target": label, "Variant": name,
            "Accuracy": round(acc, 3), "F1_weighted": round(f1, 3),
            "ROC_AUC": round(auc, 3), "Brier": round(brier, 3),
            "Precision_Compliant": round(prec_pc[0], 3), "Recall_Compliant": round(rec_pc[0], 3),
            "F1_Compliant": round(f1_pc[0], 3),
            "Precision_NonCompliant": round(prec_pc[1], 3), "Recall_NonCompliant": round(rec_pc[1], 3),
            "F1_NonCompliant": round(f1_pc[1], 3),
            "Best_Params": search.best_params_,
        })

        fold_f1 = cross_val_score(best_model, X, y, cv=cv, scoring="f1_weighted")
        fold_acc = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")

        for i, (_, test_idx) in enumerate(cv.split(X, y), 1):
            y_true_fold = np.asarray(y)[test_idx]
            y_pred_fold = np.asarray(y_pred)[test_idx]
            prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(y_true_fold, y_pred_fold, labels=[0, 1], zero_division=0)
            fold_rows.append({
                "Target": label, "Variant": name, "Fold": i,
                "F1_weighted": fold_f1[i - 1], "Accuracy": fold_acc[i - 1],
                "Precision_NonCompliant": prec_f[1], "Recall_NonCompliant": rec_f[1], "F1_NonCompliant": f1_f[1],
                "Precision_Compliant": prec_f[0], "Recall_Compliant": rec_f[0], "F1_Compliant": f1_f[0],
            })

        print(f"    Accuracy={acc:.3f}  F1(weighted)={f1:.3f}  ROC-AUC={auc:.3f}  Brier={brier:.3f}")
        print(f"    Non-compliant class -> Precision={prec_pc[1]:.3f}  Recall={rec_pc[1]:.3f}  F1={f1_pc[1]:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    summary_df.to_csv(f"{results_dir}/{target_slug}_comparison_table.csv", index=False)
    fold_df.to_csv(f"{results_dir}/{target_slug}_fold_scores.csv", index=False)

    sig_df = _paired_significance_tests(fold_df, label, variants)
    sig_df.to_csv(f"{results_dir}/{target_slug}_significance_tests.csv", index=False)

    _save_ablation_figures(summary_df, fold_df, label, target_slug, results_dir)

    print(f"  All outputs for '{label}' saved to: {results_dir}")
    return summary_df, fold_df, sig_df


def _paired_significance_tests(fold_df, label, variants):
    rows = []
    for baseline in (name for name in variants if name != HYBRID_NAME):
        for metric in ["F1_weighted", "Accuracy", "F1_NonCompliant", "Recall_NonCompliant"]:
            hybrid_vals = fold_df[fold_df.Variant == HYBRID_NAME].sort_values("Fold")[metric].values
            baseline_vals = fold_df[fold_df.Variant == baseline].sort_values("Fold")[metric].values
            diff = hybrid_vals - baseline_vals

            t_stat, t_p = ttest_rel(hybrid_vals, baseline_vals)
            try:
                w_stat, w_p = wilcoxon(hybrid_vals, baseline_vals)
            except ValueError:
                w_stat, w_p = np.nan, np.nan

            rows.append({
                "Target": label, "Comparison": f"Hybrid vs. {baseline}", "Metric": metric,
                "Mean_Diff": round(diff.mean(), 4),
                "Paired_t_stat": round(t_stat, 3), "Paired_t_pvalue": round(t_p, 4),
                "Wilcoxon_stat": w_stat if np.isnan(w_stat) else round(w_stat, 3),
                "Wilcoxon_pvalue": w_p if np.isnan(w_p) else round(w_p, 4),
            })
    return pd.DataFrame(rows)


def _save_ablation_figures(summary_df, fold_df, label, target_slug, results_dir):
    plt.rcParams.update(PLOT_STYLE)

    # Overall metrics grouped bar chart
    metrics_to_plot = ["Accuracy", "F1_weighted", "ROC_AUC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(VARIANT_ORDER):
        vals = summary_df.set_index("Variant").loc[name, metrics_to_plot].values.astype(float)
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=VARIANT_COLORS[name], edgecolor=C_DARK, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "F1-score", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Ablation Comparison: {label}")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.6)
    plt.tight_layout(pad=2.0)
    fig.savefig(f"{results_dir}/{target_slug}_ablation_bar_chart.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


    nc_metrics = ["Precision_NonCompliant", "Recall_NonCompliant", "F1_NonCompliant"]
    x_nc = np.arange(len(nc_metrics))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(VARIANT_ORDER):
        vals = summary_df.set_index("Variant").loc[name, nc_metrics].values.astype(float)
        ax.bar(x_nc + (i - 1) * width, vals, width, label=name, color=VARIANT_COLORS[name], edgecolor=C_DARK, linewidth=0.8)
    ax.set_xticks(x_nc)
    ax.set_xticklabels(["Precision", "Recall", "F1-score"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (Non-Compliant Class)")
    ax.set_title(f"Non-Compliant Class Performance: {label}")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.6)
    plt.tight_layout(pad=2.0)
    fig.savefig(f"{results_dir}/{target_slug}_noncompliant_bar_chart.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    # Per-fold paired F1 box plot
    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = [fold_df[fold_df.Variant == name]["F1_weighted"].values for name in VARIANT_ORDER]
    bp = ax.boxplot(
        box_data, patch_artist=True, widths=0.45,
        medianprops=dict(color=C_DARK, linewidth=2),
        whiskerprops=dict(color=C_DARK, linewidth=1.2),
        capprops=dict(color=C_DARK, linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=C_MID, markeredgecolor=C_MID, alpha=0.6),
    )
    for patch, name in zip(bp["boxes"], VARIANT_ORDER):
        patch.set_facecolor(VARIANT_COLORS[name])
        patch.set_edgecolor(C_DARK)

    fold_pivot = fold_df.pivot(index="Fold", columns="Variant", values="F1_weighted")[VARIANT_ORDER]
    for _, row in fold_pivot.iterrows():
        ax.plot(range(1, len(VARIANT_ORDER) + 1), row.values, color=C_DASH, alpha=0.5, linewidth=0.8, marker="o", markersize=3, zorder=1)

    ax.set_xticks(range(1, len(VARIANT_ORDER) + 1))
    ax.set_xticklabels(["XGBoost\nOnly", "SMOTE +\nXGBoost", "OCSVM + SMOTE\n+ XGBoost"], fontsize=9)
    ax.set_ylabel("Per-Fold F1-score (weighted)")
    ax.set_title(f"Per-Fold F1-score by Model Variant: {label}")
    plt.tight_layout(pad=2.0)
    fig.savefig(f"{results_dir}/{target_slug}_fold_boxplot.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def main():
    raw_df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in raw_df.columns if c not in ("SMPId",) and c not in (cfg["column"] for cfg in TARGET_CONFIGS)]

    raw_df, base_features = prepare_features(raw_df, feature_columns)

    all_summaries, all_sig = [], []
    for cfg in TARGET_CONFIGS:
        result = run_ablation_for_target(cfg, raw_df, base_features)
        if result is not None:
            summary_df, _, sig_df = result
            all_summaries.append(summary_df)
            all_sig.append(sig_df)

    if all_summaries:
        master_summary = pd.concat(all_summaries, ignore_index=True)
        master_sig = pd.concat(all_sig, ignore_index=True)
        master_summary.to_csv(f"{RESULTS_DIR_ROOT}/ALL_TARGETS_comparison_table.csv", index=False)
        master_sig.to_csv(f"{RESULTS_DIR_ROOT}/ALL_TARGETS_significance_tests.csv", index=False)
        print(f"\nMaster tables saved to: {RESULTS_DIR_ROOT}")
    else:
        print("\nNo targets were processed.")


if __name__ == "__main__":
    main()
