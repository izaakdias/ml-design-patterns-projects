"""Generate synthetic payment transactions with a rare positive class."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "amount",
    "transactions_last_hour",
    "merchant_risk_score",
    "distance_from_usual_km",
    "is_new_device",
    "is_cross_border",
]


def make_transactions(rows: int = 9000, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(mean=3.5, sigma=0.9, size=rows), 1.0, 2500.0)
    velocity = rng.poisson(1.0, size=rows)
    merchant_risk = rng.beta(2.0, 5.0, size=rows)
    distance = np.clip(rng.exponential(22.0, size=rows), 0.0, 1500.0)
    new_device = rng.binomial(1, 0.18, size=rows)
    cross_border = rng.binomial(1, 0.09, size=rows)
    log_odds = (
        -6.0
        + 0.38 * velocity
        + 3.0 * merchant_risk
        + 0.0025 * amount
        + 0.008 * distance
        + 1.1 * new_device
        + 1.3 * cross_border
    )
    probability = 1.0 / (1.0 + np.exp(-log_odds))
    fraud = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "transaction_id": [f"txn-{index:06d}" for index in range(rows)],
            "amount": amount.round(2),
            "transactions_last_hour": velocity,
            "merchant_risk_score": merchant_risk.round(4),
            "distance_from_usual_km": distance.round(2),
            "is_new_device": new_device,
            "is_cross_border": cross_border,
            "fraud": fraud,
        }
    )
