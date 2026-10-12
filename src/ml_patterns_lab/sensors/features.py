"""One vectorized transformation shared by batch windows and stream scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ml_patterns_lab.sensors.data import SENSOR_COLUMNS

WINDOW_SIZE = 24
FEATURE_NAMES = [
    f"{sensor}_{stat}" for sensor in SENSOR_COLUMNS for stat in ("mean", "std", "last", "range")
]


def _sensor_values(frame: pd.DataFrame) -> np.ndarray:
    if len(frame) < WINDOW_SIZE:
        raise ValueError(f"At least {WINDOW_SIZE} sensor readings are required.")
    missing = set(SENSOR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing sensor columns: {sorted(missing)}")
    values = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sensor readings must be finite.")
    return values


def _summarize(windows: np.ndarray) -> np.ndarray:
    """Input shape: windows x sensors x readings; output follows FEATURE_NAMES."""
    statistics = np.stack(
        [
            windows.mean(axis=-1),
            windows.std(axis=-1),
            windows[:, :, -1],
            np.ptp(windows, axis=-1),
        ],
        axis=-1,
    )
    return statistics.reshape(len(windows), len(FEATURE_NAMES))


def feature_row(window: pd.DataFrame) -> dict[str, float]:
    values = _sensor_values(window)[-WINDOW_SIZE:]
    features = _summarize(values.T[np.newaxis, :, :])[0]
    return dict(zip(FEATURE_NAMES, features.tolist(), strict=True))


def build_windowed_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    values = _sensor_values(frame)
    if not {"anomaly", "timestamp"}.issubset(frame.columns):
        raise ValueError("Training windows require anomaly labels and timestamps.")
    if not frame["anomaly"].isin([0, 1]).all():
        raise ValueError("Anomaly labels must be binary.")
    windows = sliding_window_view(values, WINDOW_SIZE, axis=0)
    labels = sliding_window_view(frame["anomaly"].to_numpy(), WINDOW_SIZE).max(axis=1)
    return (
        pd.DataFrame(_summarize(windows), columns=FEATURE_NAMES),
        labels.astype(np.int8),
        frame["timestamp"].iloc[WINDOW_SIZE - 1 :].reset_index(drop=True),
    )
