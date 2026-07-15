"""
Shared components of the hybrid OCSVM-XGBoost framework: the OCSVM
anomaly-feature transformer and the XGBoost hyperparameter search space.
Imported by both ocsvm_xgboost_pipeline.py (single-target training) and
ablation_study.py (multi-target ablation comparison).
"""

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

RANDOM_STATE = 42


class OCSVMAnomalyFeature(BaseEstimator, TransformerMixin):
    """
    Fits a one-class SVM on the compliant (class 0) training samples only,
    then appends a normalized anomaly score to the feature matrix. See
    Section 2.3 of the manuscript for the corresponding mathematical
    formulation.
    """

    def __init__(self, nu: float = 0.1, gamma: str = "scale"):
        self.nu = nu
        self.gamma = gamma

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        normal_mask = np.asarray(y) == 0
        if normal_mask.sum() == 0:
            raise ValueError("No class-0 (compliant) samples in this training fold.")

        self.ocsvm_ = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self.ocsvm_.fit(X_scaled[normal_mask])

        raw_scores = -self.ocsvm_.decision_function(X_scaled)
        self.score_min_ = float(raw_scores.min())
        self.score_max_ = float(raw_scores.max())
        return self

    def transform(self, X, y=None):
        X_scaled = self.scaler_.transform(X)
        raw_scores = -self.ocsvm_.decision_function(X_scaled)
        denom = (self.score_max_ - self.score_min_) + 1e-9
        normalized = np.clip((raw_scores - self.score_min_) / denom, 0, 1)
        return np.column_stack([X, normalized])

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)


def make_xgb_classifier():
    return xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        tree_method="hist",
        eval_metric="logloss",
    )


# XGBoost hyperparameter search space
XGB_PARAM_GRID = {
    "classifier__max_depth": [2, 3, 4],
    "classifier__learning_rate": [0.01, 0.05, 0.10, 0.15, 0.20],
    "classifier__n_estimators": [50, 100, 200, 300],
    "classifier__min_child_weight": [1, 3, 5, 7],
    "classifier__subsample": [0.7, 0.8, 0.9],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9],
    "classifier__reg_alpha": [0.1, 0.5, 1, 2, 5],
    "classifier__reg_lambda": [1, 1.5, 2, 3, 5],
}


OCSVM_NU_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
