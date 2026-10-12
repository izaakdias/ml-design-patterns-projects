"""Churn-specific feature contract and policy."""

from sklearn.calibration import CalibratedClassifierCV

from ml_patterns_lab.churn.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ml_patterns_lab.evaluation import select_recall_threshold as choose_threshold
from ml_patterns_lab.tabular import build_classifier

__all__ = ["build_model", "choose_threshold"]


def build_model() -> CalibratedClassifierCV:
    return build_classifier(NUMERIC_FEATURES, CATEGORICAL_FEATURES, seed=41)
