"""Generate a reproducible synthetic customer dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets_90d",
    "plan",
    "region",
]
NUMERIC_FEATURES = ["tenure_months", "monthly_spend", "support_tickets_90d"]
CATEGORICAL_FEATURES = ["plan", "region"]


def make_dataset(rows: int = 4000, seed: int = 41) -> pd.DataFrame:
    """Create synthetic subscription records and a probabilistic churn label."""
    rng = np.random.default_rng(seed)
    plans = rng.choice(["essencial", "conforto", "premium"], size=rows, p=[0.46, 0.34, 0.20])
    regions = rng.choice(["norte", "sul", "leste", "oeste"], size=rows)
    tenure = rng.integers(1, 73, size=rows)
    plan_price = np.select(
        [plans == "essencial", plans == "conforto", plans == "premium"],
        [38.0, 68.0, 112.0],
    )
    monthly_spend = np.clip(rng.normal(plan_price, 11.0), 20.0, 160.0)
    support_tickets = rng.poisson(0.55 + 0.75 * (monthly_spend > 90.0))

    log_odds = (
        -1.8
        + 0.028 * (monthly_spend - 65.0)
        + 0.48 * support_tickets
        - 0.030 * (tenure - 24.0)
        + 0.18 * (plans == "essencial")
    )
    churn_probability = 1.0 / (1.0 + np.exp(-log_odds))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_spend": monthly_spend.round(2),
            "support_tickets_90d": support_tickets,
            "plan": plans,
            "region": regions,
            "churn": churn,
        }
    )
