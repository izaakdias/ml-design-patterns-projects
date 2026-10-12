"""Backtest recursive forecasts without reading future observations."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml_patterns_lab.demand.forecast import forecast_horizon


def evaluate_recursive(model, series: pd.Series, test_start: int, horizon: int = 24) -> dict:
    if test_start < 24 or test_start >= len(series) or not 1 <= horizon <= 48:
        raise ValueError("Invalid backtest origin or horizon.")
    actual, predicted, baseline = [], [], []
    for origin in range(test_start, len(series), horizon):
        count = min(horizon, len(series) - origin)
        history = series.iloc[:origin].to_list()
        forecasts = forecast_horizon(model, history, series.index[origin - 1], count)
        actual.extend(series.iloc[origin : origin + count].to_list())
        predicted.extend(point["demand_kwh"] for point in forecasts)
        baseline.extend(np.resize(history[-24:], count).tolist())
    return {
        "horizon_hours": horizon,
        "model_mae_kwh": round(float(mean_absolute_error(actual, predicted)), 3),
        "seasonal_baseline_mae_kwh": round(float(mean_absolute_error(actual, baseline)), 3),
        "scored_hours": len(actual),
    }
