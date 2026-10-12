"""Confidence-gated cascade with an explicit human-review neutral class."""

from __future__ import annotations

import numpy as np

HUMAN_REVIEW = "revisao_humana"


def route_ticket(
    fast_model,
    expert_model,
    text: str,
    fast_threshold: float | None,
    human_review_threshold: float | None = 0.52,
) -> dict[str, str | float | bool]:
    """Use a fast model first, then a stronger model, then abstain if uncertain."""
    if not text.strip():
        raise ValueError("text cannot be blank.")
    values = [text]
    # Unseen vocabulary must not route from a classifier's class prior alone.
    if fast_model.named_steps["text_features"].transform(values).nnz == 0:
        return {
            "intent": HUMAN_REVIEW,
            "confidence": 0.0,
            "stage": "neutral",
            "human_review_required": True,
        }
    fast_probabilities = fast_model.predict_proba(values)[0]
    fast_index = int(np.argmax(fast_probabilities))
    fast_confidence = float(fast_probabilities[fast_index])
    if fast_threshold is not None and fast_confidence >= fast_threshold:
        return {
            "intent": str(fast_model.classes_[fast_index]),
            "confidence": fast_confidence,
            "stage": "fast",
            "human_review_required": False,
        }

    expert_probabilities = expert_model.predict_proba(values)[0]
    expert_index = int(np.argmax(expert_probabilities))
    expert_confidence = float(expert_probabilities[expert_index])
    if human_review_threshold is None or expert_confidence < human_review_threshold:
        return {
            "intent": HUMAN_REVIEW,
            "confidence": expert_confidence,
            "stage": "neutral",
            "human_review_required": True,
        }
    return {
        "intent": str(expert_model.classes_[expert_index]),
        "confidence": expert_confidence,
        "stage": "expert",
        "human_review_required": False,
    }
