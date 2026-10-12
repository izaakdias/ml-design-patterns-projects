"""Train, compare, evaluate, and version the churn-prioritization model."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from ml_patterns_lab.churn.data import FEATURES, make_dataset
from ml_patterns_lab.churn.model import build_model, choose_threshold
from ml_patterns_lab.evaluation import binary_metrics, slice_metrics
from ml_patterns_lab.shared import save_bundle
from ml_patterns_lab.tabular import global_logit_factors


def train() -> tuple[Path, dict]:
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

    model = validation_model
    probabilities = model.predict_proba(X_test)[:, 1]
    predicted = (probabilities >= threshold).astype(int)

    heuristic = (
        (X_test["support_tickets_90d"].to_numpy() >= 3)
        | ((X_test["tenure_months"].to_numpy() <= 4) & (X_test["monthly_spend"].to_numpy() >= 90))
    ).astype(int)
    metrics = binary_metrics(y_test, predicted, probabilities)
    metrics["heuristic_benchmark"] = binary_metrics(y_test, heuristic, heuristic.astype(float))
    metrics["group_slices_by_region"] = slice_metrics(
        X_test["region"].to_numpy(), y_test, predicted
    )
    metrics["threshold_selected_on_validation"] = round(threshold, 4)

    global_factors = global_logit_factors(model)
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
