"""HTTP contracts for customer-success prioritization."""

from typing import Literal

from pydantic import Field

from ml_patterns_lab.churn.inference import score_customer
from ml_patterns_lab.serving import (
    APIModel,
    ModelDependency,
    Probability,
    VersionedResponse,
    create_app,
)


class ChurnRequest(APIModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(gt=0, le=1000)
    support_tickets_90d: int = Field(ge=0, le=100)
    plan: Literal["essencial", "conforto", "premium"]
    region: Literal["norte", "sul", "leste", "oeste"]


class ChurnResponse(VersionedResponse):
    churn_probability: Probability
    threshold: Probability
    recommendation: Literal["review_for_retention", "no_alert"]


app = create_app(
    "churn",
    title="Customer-success churn prioritization",
    description="Calibrated synthetic customer-risk model with a review policy.",
)


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest, artifact: ModelDependency) -> dict:
    result = score_customer(artifact.bundle, payload.model_dump(), artifact.metadata["threshold"])
    return {**result, "model_version": artifact.run_id}
