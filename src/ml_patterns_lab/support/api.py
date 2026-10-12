"""HTTP contracts for Portuguese-language ticket triage."""

from typing import Literal

from pydantic import Field

from ml_patterns_lab.serving import (
    APIModel,
    ModelDependency,
    Probability,
    VersionedResponse,
    create_app,
)
from ml_patterns_lab.support.cascade import route_ticket


class TicketRequest(APIModel):
    text: str = Field(min_length=8, max_length=2000)


class TicketResponse(VersionedResponse):
    intent: Literal["cobranca", "tecnico", "conta", "entrega", "revisao_humana"]
    confidence: Probability
    stage: Literal["fast", "expert", "neutral"]
    human_review_required: bool


app = create_app(
    "support",
    title="Support-ticket triage",
    description="Synthetic Portuguese tickets evaluated with disjoint phrase families.",
)


@app.post("/predict", response_model=TicketResponse)
def predict(payload: TicketRequest, artifact: ModelDependency) -> dict:
    result = route_ticket(
        artifact.bundle["fast"],
        artifact.bundle["expert"],
        payload.text,
        artifact.metadata["fast_threshold"],
        artifact.metadata["human_review_threshold"],
    )
    return {**result, "model_version": artifact.run_id}
