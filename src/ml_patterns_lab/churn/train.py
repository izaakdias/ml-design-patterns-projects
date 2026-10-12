"""Train, compare, evaluate, and version the churn-prioritization model."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml_patterns_lab.churn.data import FEATURES, make_dataset
from ml_patterns_lab.churn.model import build_model, choose_threshold
from ml_patterns_lab.shared import save_bundle


def _binary_metrics(labels: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
    }


def _region_metrics(frame, labels: np.ndarray, predicted: np.ndarray) -> dict:
    result = {}
    for region in sorted(frame["region"].unique()):
        mask = frame["region"].to_numpy() == region
        group_labels = labels[mask]
        group_predicted = predicted[mask]
        negatives = max(int((group_labels == 0).sum()), 1)
        positives = max(int((group_labels == 1).sum()), 1)
        result[str(region)] = {
            "n": int(mask.sum()),
            "positive_rate": round(float(group_predicted.mean()), 4),
            "recall": round(float(((group_predicted == 1) & (group_labels == 1)).sum() / positives), 4),
            "false_positive_rate": round(
                float(((group_predicted == 1) & (group_labels == 0)).sum() / negatives),
                4,
            ),
        }
    return result


def main() -> None:
    data = make_dataset()
    X = data[FEATURES]
    y = data["churn"].to_numpy()
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=41,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.25,
        random_state=42,
        stratify=y_train_valid,
    )

    validation_model = build_model()
    validation_model.fit(X_train, y_train)
    validation_probabilities = validation_model.predict_proba(X_validation)[:, 1]
    threshold = choose_threshold(y_validation, validation_probabilities)

    model = build_model()
    model.fit(X_train_valid, y_train_valid)
    probabilities = model.predict_proba(X_test)[:, 1]
    predicted = (probabilities >= threshold).astype(int)

    heuristic = (
        (X_test["support_tickets_90d"].to_numpy() >= 3)
        | (
            (X_test["tenure_months"].to_numpy() <= 4)
            & (X_test["monthly_spend"].to_numpy() >= 90)
        )
    ).astype(int)
    metrics = _binary_metrics(y_test, predicted, probabilities)
    metrics["heuristic_benchmark"] = _binary_metrics(y_test, heuristic, heuristic.astype(float))
    metrics["group_slices_by_region"] = _region_metrics(X_test, y_test, predicted)
    metrics["threshold_selected_on_validation"] = round(threshold, 4)

    classifier = model.named_steps["classifier"]
    names = model.named_steps["transform"].get_feature_names_out()
    coefficients = classifier.coef_[0]
    order = np.argsort(np.abs(coefficients))[::-1][:8]
    global_factors = [
        {"feature": str(names[index]), "logit_coefficient": round(float(coefficients[index]), 4)}
        for index in order
    ]
    version_path, metadata = save_bundle(
        "churn",
        model,
        {
            "threshold": threshold,
            "features": FEATURES,
            "metrics": metrics,
            "global_logit_factors": global_factors,
            "data_seed": 41,
            "data_rows": int(len(data)),
        },
    )
    print(json.dumps({"artifact": str(version_path), "metadata": metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
