"""Replay recent synthetic telemetry through a stateful rolling window."""

from __future__ import annotations

import json

from ml_patterns_lab.sensors.data import make_sensor_data
from ml_patterns_lab.sensors.inference import score_window
from ml_patterns_lab.sensors.stream import RollingWindow
from ml_patterns_lab.shared import load_bundle


def main() -> None:
    bundle, metadata = load_bundle("sensors")
    telemetry = make_sensor_data()
    state = RollingWindow(metadata["window_size"])
    alerts = []
    replay = telemetry.tail(600)
    columns = ["temperature_c", "vibration_mm_s", "pressure_kpa"]
    for reading in replay[columns].to_dict(orient="records"):
        window = state.append(reading)
        if window is not None:
            alerts.append(score_window(bundle, metadata, window))
    print(
        json.dumps(
            {
                "stream_readings": int(len(replay)),
                "scored_windows": len(alerts),
                "alert_count": sum(item["anomaly_flag"] for item in alerts),
                "model_version": metadata["run_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
