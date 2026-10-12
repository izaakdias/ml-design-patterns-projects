"""Two-stage visual prediction with a manual-review fallback."""

from __future__ import annotations

import numpy as np

from ml_patterns_lab.vision.data import IMAGE_SIZE, quantize_images


def route_image(
    bundle: dict,
    pixels: list[float] | np.ndarray,
    fast_confidence_threshold: float,
    expert_review_threshold: float,
) -> dict[str, str | float | bool]:
    image = np.asarray(pixels, dtype=np.float32)
    if image.size != IMAGE_SIZE:
        raise ValueError(f"Expected {IMAGE_SIZE} grayscale pixel values.")
    if np.any(image < 0.0) or np.any(image > 1.0):
        raise ValueError("Pixel values must be between 0 and 1.")
    row = quantize_images(image.reshape(1, IMAGE_SIZE))[0].reshape(1, IMAGE_SIZE)

    fast_probabilities = bundle["fast"].predict_proba(row)[0]
    fast_index = int(np.argmax(fast_probabilities))
    fast_confidence = float(fast_probabilities[fast_index])
    if fast_confidence >= fast_confidence_threshold:
        return {
            "inspection": "defect" if fast_index == 1 else "clear",
            "confidence": fast_confidence,
            "stage": "fast",
            "manual_review_required": False,
        }

    expert_probabilities = bundle["expert"].predict_proba(row)[0]
    expert_index = int(np.argmax(expert_probabilities))
    expert_confidence = float(expert_probabilities[expert_index])
    if expert_confidence < expert_review_threshold:
        return {
            "inspection": "manual_review",
            "confidence": expert_confidence,
            "stage": "review",
            "manual_review_required": True,
        }
    return {
        "inspection": "defect" if expert_index == 1 else "clear",
        "confidence": expert_confidence,
        "stage": "expert",
        "manual_review_required": False,
    }
