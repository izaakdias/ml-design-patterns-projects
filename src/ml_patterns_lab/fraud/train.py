"""Train a rare-event model and report its two-stage review policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ml_patterns_lab.evaluation import binary_metrics
from ml_patterns_lab.fraud.data import FEATURES, make_transactions
from ml_patterns_lab.fraud.model import (
    build_model,
    choose_review_threshold,
    choose_urgent_threshold,
)
from ml_patterns_lab.shared import save_bundle


def _policy_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    return {
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 4),
        "review_rate": round(float(np.mean(predicted)), 4),
    }


def train() -> tuple[Path, dict]:
    data = make_transactions()
    X = data[FEATURES]
    y = data["fraud"].to_numpy()
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=43,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.25,
        random_state=44,
        stratify=y_train_valid,
    )

    validation_model = build_model()
    validation_model.fit(X_train, y_train)
    validation_probabilities = validation_model.predict_proba(X_validation)[:, 1]
    review_threshold = choose_review_threshold(y_validation, validation_probabilities)
    urgent_threshold = choose_urgent_threshold(
        y_validation,
        validation_probabilities,
        review_threshold,
    )

    model = validation_model
    probabilities = model.predict_proba(X_test)[:, 1]
    baseline = (
        (X_test["merchant_risk_score"].to_numpy() >= 0.82)
        | (X_test["transactions_last_hour"].to_numpy() >= 5)
        | (X_test["is_cross_border"].to_numpy() == 1)
    )
    metrics = {
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "fraud_prevalence": round(float(np.mean(y_test)), 4),
        "model_at_review_threshold": _policy_metrics(y_test, probabilities, review_threshold),
        "heuristic_benchmark": {
            "precision": round(float(precision_score(y_test, baseline, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, baseline, zero_division=0)), 4),
            "review_rate": round(float(np.mean(baseline)), 4),
        },
        "urgent_review_rate": round(float(np.mean(probabilities >= urgent_threshold)), 4)
        if urgent_threshold is not None
        else 0.0,
        "calibrated_model": binary_metrics(
            y_test, probabilities >= review_threshold, probabilities
        ),
        "thresholds_selected_on_validation": {
            "review": round(review_threshold, 4),
            "urgent_review": round(urgent_threshold, 4) if urgent_threshold is not None else None,
        },
    }
    version_path, metadata = save_bundle(
        "fraud",
        {
            "model": model,
            "review_threshold": review_threshold,
            "urgent_threshold": urgent_threshold,
        },
        {
            "features": FEATURES,
            "metrics": metrics,
            "data_seed": 43,
            "data_rows": int(len(data)),
            "split_rows": {
                "train": len(X_train),
                "validation": len(X_validation),
                "test": len(X_test),
            },
            "probability_calibration": "sigmoid_out_of_fold_training_only",
        },
    )
    return version_path, metadata


def main() -> None:
    version_path, metadata = train()
    print(
        json.dumps(
            {"artifact": str(version_path), "metadata": metadata}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
