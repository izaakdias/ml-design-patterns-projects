"""HTTP contracts for keyed transaction-risk triage."""

from typing import Literal

from pydantic import Field

from ml_patterns_lab.fraud.inference import score_transaction
from ml_patterns_lab.serving import (
    APIModel,
    ModelDependency,
    Probability,
    VersionedResponse,
    create_app,
)


class TransactionRequest(APIModel):
    transaction_id: str = Field(min_length=1, max_length=80)
    amount: float = Field(gt=0, le=100000)
    transactions_last_hour: int = Field(ge=0, le=1000)
    merchant_risk_score: Probability
    distance_from_usual_km: float = Field(ge=0, le=50000)
    is_new_device: bool
    is_cross_border: bool


class TransactionResponse(VersionedResponse):
    transaction_id: str
    fraud_score: Probability
    action: Literal["urgent_human_review", "human_review", "standard_queue"]


app = create_app(
    "fraud",
    title="Transaction-risk triage",
    description="Calibrated synthetic rare-event model with validation-selected review bands.",
)


@app.post("/predict", response_model=TransactionResponse)
def predict(payload: TransactionRequest, artifact: ModelDependency) -> dict:
    result = score_transaction(artifact.bundle, payload.model_dump(), artifact.metadata["features"])
    return {**result, "model_version": artifact.run_id}
