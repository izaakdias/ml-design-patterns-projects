"""Feature transformations and threshold selection for churn prioritization."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from ml_patterns_lab.churn.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


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
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1200,
        random_state=41,
    )
    return Pipeline([("transform", transform), ("classifier", classifier)])


def choose_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float = 0.70,
) -> float:
    """Choose one validation threshold that preserves a minimum positive recall."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.10, 0.90, 81):
        predicted = probabilities >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predicted,
            average="binary",
            zero_division=0,
        )
        if recall >= minimum_recall and (f1 > best_f1 or (f1 == best_f1 and precision > 0)):
            best_threshold = float(threshold)
            best_f1 = float(f1)
    return best_threshold
