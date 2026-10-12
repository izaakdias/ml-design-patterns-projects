"""Validate, split, train, evaluate, explain, and register a model run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from ml_patterns_lab.evaluation import binary_metrics, select_recall_threshold, slice_metrics
from ml_patterns_lab.mlops.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    GROUP_COLUMN,
    NUMERIC_FEATURES,
    TARGET,
    make_dataset,
)
from ml_patterns_lab.shared import save_bundle
from ml_patterns_lab.tabular import build_classifier, global_logit_factors

PIPELINE_STAGES = [
    "generate_data",
    "validate_schema_and_ranges",
    "repeatable_train_validation_test_split",
    "fit_transformation_and_classifier",
    "select_threshold_on_validation",
    "evaluate_metrics_and_plant_slices",
    "summarize_global_coefficients",
    "register_immutable_model_version",
]


def validate_dataset(data) -> dict:
    required = set(FEATURES + [GROUP_COLUMN, TARGET])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty or not np.isfinite(data[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("A nonempty dataset with finite numeric values is required.")
    if not data["machine_type"].isin(["pump", "compressor", "conveyor"]).all():
        raise ValueError("Unknown machine_type.")
    if not data[GROUP_COLUMN].isin(["plant_a", "plant_b", "plant_c"]).all():
        raise ValueError("Unknown plant.")
    if data[list(required)].isnull().any().any():
        raise ValueError("Synthetic training data must not contain missing required values.")
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("maintenance_due must be binary.")
    if not data["hours_since_service"].between(0, 20000).all():
        raise ValueError("hours_since_service is outside the allowed range.")
    if not data["temperature_c"].between(-10, 150).all():
        raise ValueError("temperature_c is outside the allowed range.")
    if not data["vibration_mm_s"].between(0, 30).all():
        raise ValueError("vibration_mm_s is outside the allowed range.")
    if not data["load_percent"].between(0, 100).all():
        raise ValueError("load_percent is outside the allowed range.")
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "positive_label_rate": round(float(data[TARGET].mean()), 4),
        "plants": sorted(data[GROUP_COLUMN].astype(str).unique().tolist()),
    }


def run_pipeline(seed: int = 81) -> tuple[Path, dict]:
    data = make_dataset(seed=seed)
    quality_summary = validate_dataset(data)
    X = data[FEATURES]
    y = data[TARGET].to_numpy()
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.25,
        random_state=seed + 1,
        stratify=y_train_valid,
    )

    validation_model = build_classifier(NUMERIC_FEATURES, CATEGORICAL_FEATURES, seed=seed)
    validation_model.fit(X_train, y_train)
    validation_probability = validation_model.predict_proba(X_validation)[:, 1]
    threshold = select_recall_threshold(y_validation, validation_probability)

    model = validation_model
    probability = model.predict_proba(X_test)[:, 1]
    predicted = probability >= threshold
    baseline = (
        (X_test["hours_since_service"].to_numpy() >= 1350.0)
        | (X_test["vibration_mm_s"].to_numpy() >= 4.8)
        | (X_test["temperature_c"].to_numpy() >= 91.0)
    )
    test_groups = data.loc[X_test.index, [GROUP_COLUMN]]
    metrics = {
        "model": binary_metrics(y_test, predicted, probability),
        "heuristic_benchmark": {
            "precision": round(float(precision_score(y_test, baseline, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, baseline, zero_division=0)), 4),
        },
        "plant_slices": slice_metrics(test_groups[GROUP_COLUMN].to_numpy(), y_test, predicted),
        "threshold_selected_on_validation": round(threshold, 4),
    }
    global_factors = global_logit_factors(model)
    bundle = {"model": model, "threshold": threshold}
    return save_bundle(
        "mlops",
        bundle,
        {
            "metrics": metrics,
            "data_quality": quality_summary,
            "global_logit_factors": global_factors,
            "features": FEATURES,
            "random_seed": seed,
            "split_rows": {
                "train": len(X_train),
                "validation": len(X_validation),
                "test": len(X_test),
            },
            "probability_calibration": "sigmoid_out_of_fold_training_only",
        },
        documents={
            "workflow.json": {
                "pipeline_stages": PIPELINE_STAGES,
                "data_quality": quality_summary,
                "metrics": metrics,
                "global_logit_factors": global_factors,
            }
        },
    )
