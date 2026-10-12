"""Train a chronological forecasting model against a seasonal benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from ml_patterns_lab.demand.data import make_series
from ml_patterns_lab.demand.evaluation import evaluate_recursive
from ml_patterns_lab.demand.features import FEATURE_NAMES, build_supervised
from ml_patterns_lab.shared import save_bundle


def train() -> tuple[Path, dict]:
    series = make_series()
    features, target = build_supervised(series)
    split = int(len(features) * 0.80)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target[:split], target[split:]

    model = HistGradientBoostingRegressor(
        max_iter=140,
        learning_rate=0.08,
        l2_regularization=0.2,
        random_state=41,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    seasonal_baseline = X_test["lag_24"].to_numpy()
    metrics = {
        "model_mae_kwh": round(float(mean_absolute_error(y_test, predictions)), 3),
        "seasonal_baseline_mae_kwh": round(
            float(mean_absolute_error(y_test, seasonal_baseline)), 3
        ),
        "model_r2": round(float(r2_score(y_test, predictions)), 4),
        "chronological_train_rows": int(len(X_train)),
        "chronological_test_rows": int(len(X_test)),
        "recursive_24_hour_backtest": evaluate_recursive(model, series, split + 24),
    }
    version_path, metadata = save_bundle(
        "demand",
        model,
        {
            "features": FEATURE_NAMES,
            "metrics": metrics,
            "data_seed": 41,
            "series_rows": int(len(series)),
            "frequency": "hourly",
        },
    )
    return version_path, metadata


def main() -> None:
    version_path, metadata = train()
    print(
        json.dumps(
            {"artifact": str(version_path), "metadata": metadata}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
