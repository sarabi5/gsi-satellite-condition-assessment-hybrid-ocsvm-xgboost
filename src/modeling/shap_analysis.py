"""
SHAP feature-importance analysis for a fitted hybrid OCSVM-SMOTE-XGBoost
"""

import matplotlib.pyplot as plt
import shap


def plot_shap_summary(best_model, X, output_path, title="SHAP Feature Importance"):
    ocsvm_transformer = best_model.named_steps["ocsvm_feature"]
    xgb_model = best_model.named_steps["classifier"]

    X_augmented = ocsvm_transformer.transform(X)
    feature_names = list(X.columns) + ["Anomaly_Score"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_augmented)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_augmented, feature_names=feature_names, show=False, plot_type="dot", color_bar=True)
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.show()
    print(f"SHAP figure saved: {output_path}")

    return shap_values


if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be imported and called with a fitted "
        "model and feature matrix (see plot_shap_summary docstring), or "
        "adapted to load a saved model before running standalone."
    )
