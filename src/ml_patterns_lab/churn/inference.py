"""Customer scoring independent of transport and artifact lookup."""

import pandas as pd


def score_customer(model, features: dict, threshold: float) -> dict:
    probability = float(model.predict_proba(pd.DataFrame([features]))[0, 1])
    return {
        "churn_probability": probability,
        "threshold": threshold,
        "recommendation": "review_for_retention" if probability >= threshold else "no_alert",
    }
