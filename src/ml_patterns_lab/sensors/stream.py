"""Stateful rolling-window wrapper for a sequential telemetry stream."""

from __future__ import annotations

from collections import deque

from ml_patterns_lab.sensors.features import WINDOW_SIZE


class RollingWindow:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self._readings = deque(maxlen=window_size)

    def append(self, reading: dict) -> list[dict] | None:
        self._readings.append(reading)
        if len(self._readings) < self._readings.maxlen:
            return None
        return list(self._readings)
