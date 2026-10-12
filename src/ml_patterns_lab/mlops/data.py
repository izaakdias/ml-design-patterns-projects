"""Generate synthetic equipment readings and maintenance-review labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "hours_since_service",
    "temperature_c",
    "vibration_mm_s",
    "load_percent",
    "machine_type",
]
NUMERIC_FEATURES = [
    "hours_since_service",
    "temperature_c",
    "vibration_mm_s",
    "load_percent",
]
CATEGORICAL_FEATURES = ["machine_type"]
GROUP_COLUMN = "plant"
TARGET = "maintenance_due"


def make_dataset(rows: int = 5000, seed: int = 81) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    machine_type = rng.choice(["pump", "compressor", "conveyor"], size=rows)
    plant = rng.choice(["plant_a", "plant_b", "plant_c"], size=rows)
    hours = rng.uniform(0.0, 1800.0, size=rows)
    temperature = np.clip(rng.normal(68.0, 10.0, size=rows), 30.0, 115.0)
    vibration = np.clip(rng.normal(3.0, 0.75, size=rows), 0.3, 9.0)
    load = np.clip(rng.normal(62.0, 18.0, size=rows), 5.0, 100.0)
    type_effect = np.select(
        [machine_type == "pump", machine_type == "compressor"],
        [0.10, 0.30],
        default=-0.08,
    )
    log_odds = (
        -2.25
        + 0.0022 * (hours - 700.0)
        + 0.055 * (temperature - 68.0)
        + 0.72 * (vibration - 3.0)
        + 0.018 * (load - 62.0)
        + type_effect
    )
    probability = 1.0 / (1.0 + np.exp(-log_odds))
    maintenance_due = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "hours_since_service": hours.round(1),
            "temperature_c": temperature.round(2),
            "vibration_mm_s": vibration.round(3),
            "load_percent": load.round(2),
            "machine_type": machine_type,
            GROUP_COLUMN: plant,
            TARGET: maintenance_due,
        }
    )
