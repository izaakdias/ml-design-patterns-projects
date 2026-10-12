"""Train and evaluate a text-routing cascade against a simple benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ml_patterns_lab.evaluation import select_confidence_threshold
from ml_patterns_lab.shared import save_bundle
from ml_patterns_lab.support.cascade import HUMAN_REVIEW, route_ticket
from ml_patterns_lab.support.data import make_dataset
from ml_patterns_lab.support.split import split_phrase_families


def _pipeline(classifier):
    return Pipeline(
        [
            (
                "text_features",
                TfidfVectorizer(ngram_range=(1, 2), max_features=2500, strip_accents="unicode"),
            ),
            ("classifier", classifier),
        ]
    )


def train() -> tuple[Path, dict]:
    data = make_dataset()
    training, validation, test = split_phrase_families(data)
    X_train, y_train = training["text"], training["intent"]
    X_validation, y_validation = validation["text"], validation["intent"]
    X_test, y_test = test["text"], test["intent"]

    fast_model = _pipeline(MultinomialNB(alpha=0.25))
    expert_model = _pipeline(LogisticRegression(max_iter=1200, class_weight="balanced"))
    fast_model.fit(X_train, y_train)
    expert_model.fit(X_train, y_train)
    validation_confidence = fast_model.predict_proba(X_validation).max(axis=1)
    fast_threshold = select_confidence_threshold(
        y_validation.to_numpy(),
        fast_model.predict(X_validation),
        validation_confidence,
        minimum_accuracy=0.90,
    )
    human_review_threshold = select_confidence_threshold(
        y_validation.to_numpy(),
        expert_model.predict(X_validation),
        expert_model.predict_proba(X_validation).max(axis=1),
        minimum_accuracy=0.80,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    routed = [
        route_ticket(fast_model, expert_model, text, fast_threshold, human_review_threshold)
        for text in X_test.to_list()
    ]
    handled = [item["intent"] != HUMAN_REVIEW for item in routed]
    handled_count = int(sum(handled))
    handled_correct = sum(
        item["intent"] == label
        for item, label, is_handled in zip(routed, y_test.to_list(), handled, strict=True)
        if is_handled
    )
    metrics = {
        "dummy_benchmark_accuracy": round(float(accuracy_score(y_test, baseline_predictions)), 4),
        "dummy_benchmark_macro_f1": round(
            float(f1_score(y_test, baseline_predictions, average="macro", zero_division=0)),
            4,
        ),
        "cascade_coverage": round(float(np.mean(handled)), 4),
        "accuracy_when_routed": round(float(handled_correct / handled_count), 4)
        if handled_count
        else None,
        "fast_path_share": round(
            float(sum(item["stage"] == "fast" for item in routed) / len(routed)),
            4,
        ),
        "expert_path_share": round(
            float(sum(item["stage"] == "expert" for item in routed) / len(routed)),
            4,
        ),
        "human_review_share": round(
            float(sum(item["stage"] == "neutral" for item in routed) / len(routed)),
            4,
        ),
        "fast_threshold_selected_on_validation": round(fast_threshold, 4)
        if fast_threshold is not None
        else None,
    }
    version_path, metadata = save_bundle(
        "support",
        {"fast": fast_model, "expert": expert_model},
        {
            "fast_threshold": fast_threshold,
            "human_review_threshold": human_review_threshold,
            "metrics": metrics,
            "intents": sorted(data["intent"].unique().tolist()),
            "data_rows": int(len(data)),
            "split_strategy": "disjoint_phrase_families_per_intent",
            "split_rows": {
                "train": len(training),
                "validation": len(validation),
                "test": len(test),
            },
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
