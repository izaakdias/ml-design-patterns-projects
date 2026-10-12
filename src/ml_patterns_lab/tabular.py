"""Calibrated tabular classifiers and coefficient explanations."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_classifier(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    seed: int,
) -> CalibratedClassifierCV:
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    estimator = Pipeline(
        [
            ("transform", transform),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1200, random_state=seed),
            ),
        ]
    )
    # Calibration uses training data only; threshold selection has its own holdout.
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=3, ensemble=False)


def global_logit_factors(model: CalibratedClassifierCV, limit: int = 8) -> list[dict]:
    """Explain underlying linear logits, not the subsequent calibration transform."""
    estimator = model.calibrated_classifiers_[0].estimator
    names = estimator.named_steps["transform"].get_feature_names_out()
    coefficients = estimator.named_steps["classifier"].coef_[0]
    order = np.argsort(np.abs(coefficients))[::-1][:limit]
    return [
        {"feature": str(names[index]), "logit_coefficient": round(float(coefficients[index]), 4)}
        for index in order
    ]
