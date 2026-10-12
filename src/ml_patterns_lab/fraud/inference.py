"""Transaction scoring with an explicit optional urgent-review band."""

import pandas as pd

from ml_patterns_lab.fraud.model import review_action


def score_transaction(bundle: dict, transaction: dict, features: list[str]) -> dict:
    row = pd.DataFrame([{name: transaction[name] for name in features}])
    score = float(bundle["model"].predict_proba(row)[0, 1])
    return {
        "transaction_id": transaction["transaction_id"],
        "fraud_score": score,
        "action": review_action(score, bundle["review_threshold"], bundle["urgent_threshold"]),
    }
