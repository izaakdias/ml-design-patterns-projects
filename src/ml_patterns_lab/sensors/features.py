"""Create the same rolling-window summary features for batch and stream use."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_patterns_lab.sensors.data import SENSOR_COLUMNS


WINDOW_SIZE = 24
FEATURE_NAMES = [
    f"{sensor}_{stat}"
    for sensor in SENSOR_COLUMNS
    for stat in ("mean", "std", "last", "range")
]


def feature_row(window: pd.DataFrame) -> dict[str, float]:
    if len(window) < WINDOW_SIZE:
        raise ValueError(f"At least {WINDOW_SIZE} sensor readings are required.")
    values = {}
    recent = window.tail(WINDOW_SIZE)
    for sensor in SENSOR_COLUMNS:
        series = recent[sensor].to_numpy(dtype=float)
        values[f"{sensor}_mean"] = float(np.mean(series))
        values[f"{sensor}_std"] = float(np.std(series))
        values[f"{sensor}_last"] = float(series[-1])
        values[f"{sensor}_range"] = float(np.max(series) - np.min(series))
    return values


def build_windowed_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rows = []
    labels = []
    timestamps = []
    for end in range(WINDOW_SIZE - 1, len(frame)):
        start = end - WINDOW_SIZE + 1
        rows.append(feature_row(frame.iloc[start : end + 1]))
        labels.append(int(frame["anomaly"].iloc[end]))
        timestamps.append(frame["timestamp"].iloc[end])
    return (
        pd.DataFrame(rows, columns=FEATURE_NAMES),
        np.asarray(labels, dtype=np.int8),
        pd.Series(timestamps, name="timestamp"),
    )
