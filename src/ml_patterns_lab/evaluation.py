"""Binary metrics and threshold policies, independent of HTTP and storage."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_binary_scores(labels: np.ndarray, probabilities: np.ndarray) -> None:
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors.")
    if not np.isin(labels, [0, 1]).all() or not (labels == 1).any():
        raise ValueError("Binary labels with at least one positive are required.")
    if (
        not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("Probabilities must be finite values between zero and one.")


def select_recall_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float = 0.70,
    *,
    objective: str = "f1",
) -> float:
    """Search observed scores; deterministic ties prefer the higher threshold."""
    validate_binary_scores(labels, probabilities)
    if not 0 < minimum_recall <= 1 or objective not in {"f1", "threshold"}:
        raise ValueError("Invalid recall target or threshold objective.")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = recall[:-1] >= minimum_recall
    if objective == "threshold":
        return float(thresholds[eligible][-1])
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.max(f1[eligible])
    return float(thresholds[eligible & np.isclose(f1, best, rtol=0, atol=1e-12)][-1])


def binary_metrics(labels: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4)
        if len(np.unique(labels)) == 2
        else None,
        "average_precision": round(float(average_precision_score(labels, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 4),
    }


def select_confidence_threshold(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    *,
    minimum_accuracy: float,
    minimum_predictions: int = 5,
) -> float | None:
    """Require both observed correctness and sample support, or disable routing."""
    if (
        labels.shape != predictions.shape
        or labels.shape != confidence.shape
        or not len(labels)
        or not np.isfinite(confidence).all()
        or np.any((confidence < 0) | (confidence > 1))
        or not 0 < minimum_accuracy <= 1
        or minimum_predictions < 1
    ):
        raise ValueError("Invalid confidence calibration inputs.")
    for threshold in np.unique(confidence):
        selected = confidence >= threshold
        if (
            selected.sum() >= minimum_predictions
            and (labels[selected] == predictions[selected]).mean() >= minimum_accuracy
        ):
            return float(threshold)
    return None


def slice_metrics(groups: np.ndarray, labels: np.ndarray, predicted: np.ndarray) -> dict:
    result = {}
    for group in sorted(set(groups)):
        mask = groups == group
        positive = labels[mask] == 1
        alerts = predicted[mask].astype(bool)
        result[str(group)] = {
            "n": int(mask.sum()),
            "recall": round(float(alerts[positive].mean()), 4) if positive.any() else None,
            "false_positive_rate": round(float(alerts[~positive].mean()), 4)
            if (~positive).any()
            else None,
            "predicted_positive_rate": round(float(alerts.mean()), 4),
        }
    return result
