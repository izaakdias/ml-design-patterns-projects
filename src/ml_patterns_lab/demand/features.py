"""One shared feature transformation for training and online inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "lag_1",
    "lag_24",
    "mean_24",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


def make_feature_row(history: list[float] | np.ndarray, target_timestamp) -> dict[str, float]:
    values = np.asarray(history, dtype=float)
    if values.size < 24:
        raise ValueError("At least 24 historical hourly values are required.")

    hour_angle = 2.0 * np.pi * target_timestamp.hour / 24.0
    weekday_angle = 2.0 * np.pi * target_timestamp.weekday() / 7.0
    return {
        "lag_1": float(values[-1]),
        "lag_24": float(values[-24]),
        "mean_24": float(np.mean(values[-24:])),
        "hour_sin": float(np.sin(hour_angle)),
        "hour_cos": float(np.cos(hour_angle)),
        "weekday_sin": float(np.sin(weekday_angle)),
        "weekday_cos": float(np.cos(weekday_angle)),
    }


def build_supervised(series: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    values = series.to_numpy(dtype=float)
    for index in range(24, len(series)):
        rows.append(make_feature_row(values[:index], series.index[index]))
    features = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return features, values[24:]
