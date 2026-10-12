"""Maintenance policy applied to the calibrated model's output."""

import pandas as pd


def score_maintenance(bundle: dict, features: dict) -> dict:
    probability = float(bundle["model"].predict_proba(pd.DataFrame([features]))[0, 1])
    threshold = float(bundle["threshold"])
    return {
        "maintenance_risk": probability,
        "threshold": threshold,
        "recommendation": "schedule_maintenance_review"
        if probability >= threshold
        else "continue_monitoring",
    }
