"""Rare-event classifier and validation-only review policies."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from ml_patterns_lab.evaluation import select_recall_threshold, validate_binary_scores
from ml_patterns_lab.fraud.data import FEATURES
from ml_patterns_lab.tabular import build_classifier


def build_model() -> CalibratedClassifierCV:
    return build_classifier(FEATURES, [], seed=43)


def choose_review_threshold(
    labels: np.ndarray, probabilities: np.ndarray, minimum_recall: float = 0.75
) -> float:
    return select_recall_threshold(labels, probabilities, minimum_recall, objective="threshold")


def choose_urgent_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    review_threshold: float,
    minimum_precision: float = 0.75,
    minimum_alerts: int = 5,
) -> float | None:
    """Disable the urgent band when evidence cannot support its precision target."""
    validate_binary_scores(labels, probabilities)
    if not 0 <= review_threshold <= 1 or not 0 < minimum_precision <= 1 or minimum_alerts < 1:
        raise ValueError("Invalid urgent-review policy.")
    for threshold in np.unique(probabilities[probabilities > review_threshold]):
        selected = probabilities >= threshold
        if selected.sum() >= minimum_alerts and labels[selected].mean() >= minimum_precision:
            return float(threshold)
    return None


def review_action(score: float, review_threshold: float, urgent_threshold: float | None) -> str:
    if urgent_threshold is not None and score >= urgent_threshold:
        return "urgent_human_review"
    return "human_review" if score >= review_threshold else "standard_queue"
