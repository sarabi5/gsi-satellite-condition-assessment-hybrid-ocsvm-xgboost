"""
Train and evaluate the OCSVM-SMOTE-XGBoost framework for a single GSI target using stratified 5-fold cross-validation.

The pipeline integrates OCSVM anomaly features, SMOTE balancing, and XGBoost classification with leakage-free fold-based training.

Outputs include performance metrics, ROC curve, calibration plot, risk scores, and site-level results.
"""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from hybrid_model import OCSVM_NU_GRID, XGB_PARAM_GRID, OCSVMAnomalyFeature, make_xgb_classifier

warnings.filterwarnings("ignore")

# --- CONFIG ---
DATA_PATH = "data/quantile_ml_ready_dataset.csv"
RESULTS_DIR = "outputs/single_target_results"

TARGET_COLUMN = "SignOfErosionId(Ranking)"
FEATURE_COLUMNS = None 

N_ITER_SEARCH = 60
N_SPLITS = 5
RANDOM_STATE = 42

PLOT_STYLE = {"font.family": "serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
C_DARK, C_MID, C_LIGHT, C_DASH = "#1a1a1a", "#555555", "#aaaaaa", "#888888"


def prepare_data(df, target_column, feature_columns):
    df = df.copy()

    # Asset age at time of inspection; negative values (data errors) set to NaN
    df["InspectionDate"] = pd.to_datetime(df["InspectionDate"])
    df["Asset_Age"] = df["InspectionDate"].dt.year - df["DATE_INSTALLED"]
    df.loc[df["Asset_Age"] < 0, "Asset_Age"] = pd.NA

    df = pd.get_dummies(df, columns=["GSI Type"], prefix="GSI", dtype=int)
    df = df.drop(columns=["DATE_INSTALLED", "InspectionDate"])

    df = df.dropna(subset=[target_column])
    # Ratings 1-2 -> compliant (0); ratings 3-4 -> non-compliant (1)
    df[target_column] = df[target_column].map({1: 0, 2: 0, 3: 1, 4: 1})

    if feature_columns is None:
        gsi_cols = [c for c in df.columns if c.startswith("GSI_")]
        exclude = {"SMPId", target_column}
        feature_columns = [c for c in df.columns if c not in exclude and c not in gsi_cols] + ["Asset_Age"] + gsi_cols

    X = df[feature_columns].copy().fillna(df[feature_columns].median())
    y = df[target_column]
    return X, y


def build_pipeline(smote_strategy):
    return Pipeline([
        ("ocsvm_feature", OCSVMAnomalyFeature()),
        ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=smote_strategy)),
        ("classifier", make_xgb_classifier()),
    ])


def run(target_column=TARGET_COLUMN, feature_columns=FEATURE_COLUMNS):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    target_slug = target_column.replace(" ", "_").replace("(", "").replace(")", "")

    df = pd.read_csv(DATA_PATH)
    X, y = prepare_data(df, target_column, feature_columns)

    class_counts = y.value_counts()
    smote_target = int(class_counts.max())
    smote_strategy = {0: smote_target, 1: smote_target}
    print(f"Target: {target_column} | Class 0: {class_counts.get(0, 0)} | Class 1: {class_counts.get(1, 0)}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_pipeline(smote_strategy)
    param_grid = {"ocsvm_feature__nu": OCSVM_NU_GRID, **XGB_PARAM_GRID}

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
    )
    search.fit(X, y)
    best_model = search.best_estimator_
    print("Best hyperparameters:", search.best_params_)

    # --- Cross-validated evaluation ---
    y_pred = cross_val_predict(best_model, X, y, cv=cv)
    y_prob = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")[:, 1]

    print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
    print(f"F1 (weighted): {f1_score(y, y_pred, average='weighted'):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y, y_prob):.3f}")
    print(f"Brier score: {brier_score_loss(y, y_prob):.3f}")
    print(classification_report(y, y_pred, target_names=["Compliant (0)", "Non-Compliant (1)"]))

    fold_scores = cross_val_score(best_model, X, y, cv=cv, scoring="f1_weighted")
    print(f"F1 by fold: {[round(s, 3) for s in fold_scores]} | mean={fold_scores.mean():.3f}")

    # --- Refit on full data for risk scoring and calibration ---
    best_model.fit(X, y)
    risk_scores = best_model.predict_proba(X)[:, 1]

    ocsvm_transformer = best_model.named_steps["ocsvm_feature"]
    xgb_model = best_model.named_steps["classifier"]
    X_augmented = ocsvm_transformer.transform(X)

    calibrated_xgb = CalibratedClassifierCV(xgb_model, method="sigmoid", cv="prefit")
    calibrated_xgb.fit(X_augmented, y)
    y_prob_calibrated = calibrated_xgb.predict_proba(X_augmented)[:, 1]
    print(f"Brier score (calibrated): {brier_score_loss(y, y_prob_calibrated):.3f}")

    _save_figures(y, y_prob, y_prob_calibrated, risk_scores, target_slug)
    _save_risk_scores(df, y, risk_scores, y_prob, y_pred, y_prob_calibrated, target_column, target_slug)


def _save_figures(y, y_prob, y_prob_calibrated, risk_scores, target_slug):
    plt.rcParams.update(PLOT_STYLE)

    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_val = roc_auc_score(y, y_prob)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=C_DARK, linewidth=2, label=f"XGBoost (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=C_DASH, linewidth=1.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.12, color=C_LIGHT)
    ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=9, loc="lower right", framealpha=0.5)
    plt.tight_layout(pad=2.0)
    fig.savefig(f"{RESULTS_DIR}/{target_slug}_roc_curve.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bp = ax.boxplot(
        [risk_scores[y == 0], risk_scores[y == 1]],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color=C_DARK, linewidth=2.5),
        whiskerprops=dict(color=C_DARK, linewidth=1.5),
        capprops=dict(color=C_DARK, linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=C_MID, markeredgecolor=C_MID, alpha=0.6),
    )
    bp["boxes"][0].set_facecolor(C_LIGHT)
    bp["boxes"][1].set_facecolor(C_MID)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Compliant\n(Class 0)", "Non-Compliant\n(Class 1)"], fontsize=9)
    ax.set(title="Risk Score by Class", ylabel="XGBoost Risk Score  P(Non-Compliance)", ylim=(0, 1))
    ax.axhline(0.5, color=C_DASH, linestyle="--", linewidth=1, alpha=0.7)
    plt.tight_layout(pad=2.0)
    fig.savefig(f"{RESULTS_DIR}/{target_slug}_risk_score_boxplot.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y, y_prob, n_bins=4)
    frac_pos_cal, mean_pred_cal = calibration_curve(y, y_prob_calibrated, n_bins=4)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(mean_pred, frac_pos, "s--", color=C_MID, linewidth=1.5, markersize=5, label="XGBoost (raw)")
    ax.plot(mean_pred_cal, frac_pos_cal, "s-", color=C_DARK, linewidth=2, markersize=6, label="XGBoost (calibrated)")
    ax.plot([0, 1], [0, 1], linestyle="--", color=C_DASH, linewidth=1.5, label="Perfect Calibration")
    ax.set(title="Calibration Curve", xlabel="Mean Predicted Probability", ylabel="Fraction of Positives", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.5)
    plt.tight_layout(pad=2.0)
    fig.savefig(f"{RESULTS_DIR}/{target_slug}_calibration_curve.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to {RESULTS_DIR}")


def _save_risk_scores(df, y, risk_scores, y_prob_cv, y_pred_cv, y_prob_calibrated, target_column, target_slug):
    output_df = df.copy()
    output_df["risk_score"] = pd.Series(risk_scores, index=y.index)
    output_df["cv_risk_score"] = pd.Series(y_prob_cv, index=y.index)
    output_df["cv_prediction"] = pd.Series(y_pred_cv, index=y.index)
    output_df["calibrated_risk_score"] = pd.Series(y_prob_calibrated, index=y.index)

    output_path = f"{RESULTS_DIR}/{target_slug}_risk_scores.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Risk scores saved to {output_path}")


if __name__ == "__main__":
    run()
