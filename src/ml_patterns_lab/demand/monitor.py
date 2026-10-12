"""A lightweight population-stability check for the synthetic demand series."""

from __future__ import annotations

import json

import numpy as np

from ml_patterns_lab.demand.data import make_series


def population_stability_index(reference, recent, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    recent = np.asarray(recent, dtype=float)
    if bins < 2 or not np.isfinite(reference).all() or not np.isfinite(recent).all():
        raise ValueError("Finite observations and at least two bins are required.")
    if reference.size == 0 or recent.size == 0:
        raise ValueError("Both reference and recent windows must contain observations.")

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        edges = np.linspace(float(reference.min()) - 1.0, float(reference.max()) + 1.0, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    epsilon = 1e-6
    reference_share = np.histogram(reference, bins=edges)[0] / reference.size
    recent_share = np.histogram(recent, bins=edges)[0] / recent.size
    reference_share = np.clip(reference_share, epsilon, None)
    recent_share = np.clip(recent_share, epsilon, None)
    return float(np.sum((recent_share - reference_share) * np.log(recent_share / reference_share)))


def main() -> None:
    series = make_series()
    reference = series.iloc[:-168].to_numpy()
    recent = series.iloc[-168:].to_numpy()
    psi = population_stability_index(reference, recent)
    print(
        json.dumps(
            {
                "reference_hours": int(reference.size),
                "recent_hours": int(recent.size),
                "psi": round(psi, 4),
                "heuristic_drift_flag": psi >= 0.20,
                "note": "PSI is a triage signal; validate its threshold and response for real data.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
