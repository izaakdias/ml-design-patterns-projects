"""Risk scoring and deviation-based hints for one sensor window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_patterns_lab.sensors.features import feature_row


def score_window(bundle: dict, metadata: dict, readings: list[dict]) -> dict:
    window = pd.DataFrame(readings)
    features = pd.DataFrame([feature_row(window)])
    risk = float(-bundle["model"].score_samples(features)[0])
    threshold = float(bundle["threshold"])
    medians = metadata["reference_median"]
    scales = metadata["reference_iqr"]
    deviations = []
    for name, value in features.iloc[0].items():
        scale = max(float(scales[name]), 1e-6)
        deviations.append(
            {
                "feature": name,
                "scaled_deviation": abs(float(value) - float(medians[name])) / scale,
            }
        )
    drivers = sorted(deviations, key=lambda item: item["scaled_deviation"], reverse=True)[:3]
    return {
        "anomaly_score": round(risk, 5),
        "anomaly_flag": risk >= threshold,
        "threshold": round(threshold, 5),
        "top_window_deviations": [
            {
                "feature": item["feature"],
                "scaled_deviation": round(float(item["scaled_deviation"]), 3),
            }
            for item in drivers
        ],
    }
