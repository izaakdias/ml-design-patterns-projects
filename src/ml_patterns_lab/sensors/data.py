"""Create synthetic sensor telemetry with short injected anomaly bursts."""

from __future__ import annotations

import numpy as np
import pandas as pd


SENSOR_COLUMNS = ["temperature_c", "vibration_mm_s", "pressure_kpa"]


def make_sensor_data(rows: int = 6000, seed: int = 71) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = np.arange(rows, dtype=float)
    temperature = 52.0 + 4.0 * np.sin(2.0 * np.pi * step / 1440.0)
    vibration = 2.6 + 0.28 * np.sin(2.0 * np.pi * step / 180.0)
    pressure = 99.0 + 1.5 * np.cos(2.0 * np.pi * step / 720.0)
    temperature += rng.normal(0.0, 0.65, size=rows)
    vibration += rng.normal(0.0, 0.09, size=rows)
    pressure += rng.normal(0.0, 0.35, size=rows)
    anomaly = np.zeros(rows, dtype=np.int8)

    starts = rng.choice(np.arange(350, rows - 30), size=14, replace=False)
    for start in starts:
        width = int(rng.integers(5, 13))
        end = min(int(start + width), rows)
        anomaly[start:end] = 1
        temperature[start:end] += rng.choice([-1.0, 1.0]) * rng.uniform(10.0, 18.0)
        vibration[start:end] += rng.uniform(2.2, 4.2)
        pressure[start:end] -= rng.uniform(7.0, 13.0)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min"),
            "temperature_c": temperature,
            "vibration_mm_s": vibration,
            "pressure_kpa": pressure,
            "anomaly": anomaly,
        }
    )
