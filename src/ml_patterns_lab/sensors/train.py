"""Fit an unsupervised detector on normal windows and calibrate alerts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from ml_patterns_lab.sensors.data import make_sensor_data
from ml_patterns_lab.sensors.features import FEATURE_NAMES, build_windowed_dataset
from ml_patterns_lab.shared import save_bundle


def train() -> tuple[Path, dict]:
    telemetry = make_sensor_data()
    # Split raw readings first so windows cannot share readings across partitions.
    train_end = int(len(telemetry) * 0.70)
    validation_end = int(len(telemetry) * 0.85)
    X_train, y_train, _ = build_windowed_dataset(telemetry.iloc[:train_end])
    X_validation, y_validation, _ = build_windowed_dataset(telemetry.iloc[train_end:validation_end])
    X_test, y_test, _ = build_windowed_dataset(telemetry.iloc[validation_end:])

    normal_training = X_train[y_train == 0]
    if normal_training.empty:
        raise ValueError("Normal training windows are required.")
    model = IsolationForest(
        n_estimators=180,
        max_samples="auto",
        contamination="auto",
        random_state=71,
        n_jobs=-1,
    )
    model.fit(normal_training)
    validation_risk = -model.score_samples(X_validation)
    normal_validation = validation_risk[y_validation == 0]
    if not normal_validation.size:
        raise ValueError("Normal validation windows are required to calibrate alerts.")
    threshold = float(np.quantile(normal_validation, 0.98))

    test_risk = -model.score_samples(X_test)
    predicted = test_risk >= threshold
    metrics = {
        "average_precision": round(float(average_precision_score(y_test, test_risk)), 4),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        "test_window_count": int(len(X_test)),
        "test_anomaly_rate": round(float(np.mean(y_test)), 4),
        "validation_false_alert_target": 0.02,
    }
    reference = normal_training
    reference_median = reference.median().to_dict()
    reference_iqr = (reference.quantile(0.75) - reference.quantile(0.25)).to_dict()
    bundle = {"model": model, "threshold": threshold}
    version_path, metadata = save_bundle(
        "sensors",
        bundle,
        {
            "features": FEATURE_NAMES,
            "metrics": metrics,
            "reference_median": {name: float(value) for name, value in reference_median.items()},
            "reference_iqr": {name: float(value) for name, value in reference_iqr.items()},
            "threshold": threshold,
            "window_size": 24,
            "data_seed": 71,
            "label_semantics": "any_anomaly_in_last_24_readings",
            "split_strategy": "chronological_raw_readings_before_windowing",
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
