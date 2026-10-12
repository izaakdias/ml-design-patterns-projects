"""Rebalancing and validation-only threshold calibration."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1200,
                    random_state=43,
                ),
            ),
        ]
    )


def choose_review_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float = 0.75,
) -> float:
    """Use the highest validation threshold that still meets the recall target."""
    candidates = np.linspace(0.01, 0.99, 99)
    eligible = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        positives = max(int((labels == 1).sum()), 1)
        recall = float(((predicted == 1) & (labels == 1)).sum() / positives)
        precision = float(((predicted == 1) & (labels == 1)).sum() / max(int(predicted.sum()), 1))
        if recall >= minimum_recall:
            eligible.append((float(threshold), precision))
    if not eligible:
        return 0.5
    return max(eligible, key=lambda item: (item[0], item[1]))[0]


def choose_urgent_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    review_threshold: float,
    minimum_precision: float = 0.75,
) -> float:
    """Reserve the urgent-review band for thresholds meeting validation precision."""
    eligible = []
    for threshold in np.linspace(max(review_threshold + 0.01, 0.02), 0.995, 100):
        predicted = probabilities >= threshold
        count = int(predicted.sum())
        if count == 0:
            continue
        precision = float(((predicted == 1) & (labels == 1)).sum() / count)
        if precision >= minimum_precision:
            eligible.append(float(threshold))
    return min(eligible) if eligible else 1.01
