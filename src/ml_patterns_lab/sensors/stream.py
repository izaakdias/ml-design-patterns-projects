"""Stateful rolling-window wrapper for a sequential telemetry stream."""

from __future__ import annotations

from collections import deque

from ml_patterns_lab.sensors.features import WINDOW_SIZE


class RollingWindow:
    def __init__(self, window_size: int = WINDOW_SIZE):
        if window_size != WINDOW_SIZE:
            raise ValueError(f"The trained feature contract requires {WINDOW_SIZE} readings.")
        self._readings: deque[dict] = deque(maxlen=window_size)

    def append(self, reading: dict) -> list[dict] | None:
        self._readings.append(dict(reading))
        if len(self._readings) < self._readings.maxlen:
            return None
        return [dict(item) for item in self._readings]
