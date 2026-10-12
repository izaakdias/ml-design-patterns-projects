"""Windowed multi-step inference for a trained demand model."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from ml_patterns_lab.demand.features import make_feature_row


def forecast_horizon(
    model,
    history: list[float],
    last_timestamp,
    horizon: int = 24,
) -> list[dict[str, float | str]]:
    """Forecast recursively, adding each prediction to the next input window."""
    values = [float(value) for value in history]
    if len(values) < 24:
        raise ValueError("At least 24 historical hourly values are required.")
    if horizon < 1 or horizon > 48:
        raise ValueError("horizon must be between 1 and 48 hours.")

    forecasts = []
    timestamp = last_timestamp
    for _ in range(horizon):
        timestamp = timestamp + timedelta(hours=1)
        features = pd.DataFrame([make_feature_row(values, timestamp)])
        demand = max(0.0, float(model.predict(features)[0]))
        forecasts.append({"timestamp": timestamp.isoformat(), "demand_kwh": round(demand, 3)})
        values.append(demand)
    return forecasts
