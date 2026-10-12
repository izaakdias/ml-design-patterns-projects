"""Generate a synthetic hourly demand series with daily and weekly cycles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_series(days: int = 120, seed: int = 41) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    step = np.arange(len(index), dtype=float)
    hour = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy()
    daily_cycle = 17.0 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0)
    weekly_cycle = 6.0 * np.cos(2.0 * np.pi * weekday / 7.0)
    gradual_change = 0.018 * step
    noise = rng.normal(0.0, 4.0, size=len(index))
    demand = 105.0 + daily_cycle + weekly_cycle + gradual_change + noise
    return pd.Series(np.maximum(demand, 1.0), index=index, name="demand_kwh")
