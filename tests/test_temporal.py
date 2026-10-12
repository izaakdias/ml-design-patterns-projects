from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ml_patterns_lab.demand.evaluation import evaluate_recursive
from ml_patterns_lab.demand.features import build_supervised
from ml_patterns_lab.demand.forecast import forecast_horizon
from ml_patterns_lab.demand.monitor import population_stability_index
from ml_patterns_lab.sensors.features import build_windowed_dataset, feature_row
from ml_patterns_lab.sensors.stream import RollingWindow
from ml_patterns_lab.support.data import make_dataset
from ml_patterns_lab.support.split import split_phrase_families


class NextValue:
    def predict(self, frame):
        return frame["lag_1"].to_numpy() + 1


def test_forecast_uses_its_previous_predictions():
    result = forecast_horizon(NextValue(), list(range(24)), datetime(2026, 1, 1), 3)
    assert [row["demand_kwh"] for row in result] == [24, 25, 26]
    assert result[-1]["timestamp"] == "2026-01-01T03:00:00"


def test_training_features_use_only_prior_observations():
    series = pd.Series(np.arange(30), index=pd.date_range("2026-01-01", periods=30, freq="h"))
    features, target = build_supervised(series)
    assert features.iloc[0]["lag_1"] == 23
    assert features.iloc[0]["lag_24"] == 0
    assert target[0] == 24
    metrics = evaluate_recursive(NextValue(), series, 24)
    assert metrics["model_mae_kwh"] == 0
    assert metrics["scored_hours"] == 6


def test_psi_catches_shift_outside_constant_reference():
    assert population_stability_index(np.ones(100), np.ones(100)) == 0
    assert population_stability_index(np.ones(100), np.full(100, 100)) > 0.2


def sensor_frame():
    return pd.DataFrame(
        {
            "temperature_c": np.arange(50) + 30,
            "vibration_mm_s": np.ones(50),
            "pressure_kpa": np.full(50, 100),
            "anomaly": [0] * 5 + [1] + [0] * 44,
            "timestamp": pd.date_range("2026-01-01", periods=50, freq="min"),
        }
    )


def test_sensor_window_stays_anomalous_until_bad_reading_exits():
    frame = sensor_frame()
    features, labels, _ = build_windowed_dataset(frame)
    assert labels[0] == 1  # Bad reading at 5 is inside [0, 23].
    assert labels[6] == 0  # First clean window is [6, 29].
    assert features.iloc[0].to_dict() == feature_row(frame.iloc[:24])


def test_stream_and_batch_features_match_and_do_not_share_mutable_state():
    frame = sensor_frame()
    stream = RollingWindow()
    result = None
    for reading in frame.iloc[:24].to_dict(orient="records"):
        result = stream.append(reading)
        reading["temperature_c"] = -1000
    assert feature_row(pd.DataFrame(result)) == feature_row(frame.iloc[:24])
    result[0]["temperature_c"] = -1000
    next_window = stream.append(frame.iloc[24].to_dict())
    assert next_window[0]["temperature_c"] != -1000


def test_windows_cannot_use_an_incompatible_size():
    with pytest.raises(ValueError):
        RollingWindow(10)


def test_batch_and_stream_transformations_match_for_every_window():
    frame = sensor_frame()
    rng = np.random.default_rng(0)
    frame["temperature_c"] += rng.normal(size=len(frame))
    features, _, _ = build_windowed_dataset(frame)
    for index in range(len(features)):
        online = feature_row(frame.iloc[index : index + 24])
        np.testing.assert_allclose(
            features.iloc[index].to_numpy(), list(online.values()), rtol=1e-12
        )


def test_support_phrase_families_are_disjoint_and_cover_dataset():
    data = make_dataset()
    partitions = split_phrase_families(data)
    groups = [set(frame["phrase_group"]) for frame in partitions]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert sum(map(len, partitions)) == len(data)
    for frame in partitions:
        assert set(frame["intent"]) == set(data["intent"])
