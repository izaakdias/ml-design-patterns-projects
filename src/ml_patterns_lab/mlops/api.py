"""HTTP contracts for the registered maintenance-risk model."""

from typing import Literal

from pydantic import Field

from ml_patterns_lab.mlops.inference import score_maintenance
from ml_patterns_lab.serving import (
    APIModel,
    ModelDependency,
    Probability,
    VersionedResponse,
    create_app,
)


class MaintenanceRequest(APIModel):
    hours_since_service: float = Field(ge=0, le=20000)
    temperature_c: float = Field(ge=-10, le=150)
    vibration_mm_s: float = Field(ge=0, le=30)
    load_percent: float = Field(ge=0, le=100)
    machine_type: Literal["pump", "compressor", "conveyor"]


class MaintenanceResponse(VersionedResponse):
    maintenance_risk: Probability
    threshold: Probability
    recommendation: Literal["schedule_maintenance_review", "continue_monitoring"]


app = create_app(
    "mlops",
    title="Maintenance-risk model service",
    description="Calibrated synthetic equipment model from the reproducible workflow.",
)


@app.post("/predict", response_model=MaintenanceResponse)
def predict(payload: MaintenanceRequest, artifact: ModelDependency) -> dict:
    return {
        **score_maintenance(artifact.bundle, payload.model_dump()),
        "model_version": artifact.run_id,
    }
