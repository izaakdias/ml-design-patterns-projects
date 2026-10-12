"""Validate, split, train, evaluate, explain, and register a model run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_patterns_lab.mlops.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    GROUP_COLUMN,
    NUMERIC_FEATURES,
    TARGET,
    make_dataset,
)
from ml_patterns_lab.shared import save_bundle


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


def build_model() -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("transform", transform),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1200, random_state=81),
            ),
        ]
    )


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.10, 0.90, 81):
        predicted = probabilities >= threshold
        positives = max(int((labels == 1).sum()), 1)
        recall = float(((predicted == 1) & (labels == 1)).sum() / positives)
        precision = float(((predicted == 1) & (labels == 1)).sum() / max(int(predicted.sum()), 1))
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        if recall >= 0.70 and f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = f1
    return best_threshold


def _slice_metrics(data, labels: np.ndarray, predicted: np.ndarray) -> dict:
    result = {}
    groups = data[GROUP_COLUMN].to_numpy()
    for group in sorted(set(groups)):
        mask = groups == group
        group_labels = labels[mask]
        group_predicted = predicted[mask]
        positives = max(int((group_labels == 1).sum()), 1)
        negatives = max(int((group_labels == 0).sum()), 1)
        result[str(group)] = {
            "n": int(mask.sum()),
            "recall": round(float(((group_predicted == 1) & (group_labels == 1)).sum() / positives), 4),
            "false_positive_rate": round(
                float(((group_predicted == 1) & (group_labels == 0)).sum() / negatives),
                4,
            ),
            "predicted_positive_rate": round(float(group_predicted.mean()), 4),
        }
    return result


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

    validation_model = build_model()
    validation_model.fit(X_train, y_train)
    validation_probability = validation_model.predict_proba(X_validation)[:, 1]
    threshold = choose_threshold(y_validation, validation_probability)

    model = build_model()
    model.fit(X_train_valid, y_train_valid)
    probability = model.predict_proba(X_test)[:, 1]
    predicted = probability >= threshold
    baseline = (
        (X_test["hours_since_service"].to_numpy() >= 1350.0)
        | (X_test["vibration_mm_s"].to_numpy() >= 4.8)
        | (X_test["temperature_c"].to_numpy() >= 91.0)
    )
    test_groups = data.loc[X_test.index, [GROUP_COLUMN]]
    metrics = {
        "model": {
            "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
            "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        },
        "heuristic_benchmark": {
            "precision": round(float(precision_score(y_test, baseline, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, baseline, zero_division=0)), 4),
        },
        "plant_slices": _slice_metrics(test_groups, y_test, predicted),
        "threshold_selected_on_validation": round(threshold, 4),
    }
    classifier = model.named_steps["classifier"]
    names = model.named_steps["transform"].get_feature_names_out()
    coefficients = classifier.coef_[0]
    top_indices = np.argsort(np.abs(coefficients))[::-1][:8]
    global_factors = [
        {"feature": str(names[index]), "logit_coefficient": round(float(coefficients[index]), 4)}
        for index in top_indices
    ]
    bundle = {"model": model, "threshold": threshold}
    version_path, metadata = save_bundle(
        "mlops",
        bundle,
        {
            "metrics": metrics,
            "data_quality": quality_summary,
            "global_logit_factors": global_factors,
            "features": FEATURES,
            "random_seed": seed,
            "split": {"train_validation_fraction": 0.80, "test_fraction": 0.20},
        },
    )
    manifest = {
        "pipeline_stages": PIPELINE_STAGES,
        "data_quality": quality_summary,
        "metrics": metrics,
        "global_logit_factors": global_factors,
        "model_run_id": metadata["run_id"],
    }
    (version_path / "workflow.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return version_path, metadata
