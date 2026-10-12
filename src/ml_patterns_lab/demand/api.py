"""HTTP contracts for recursive hourly forecasting."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field

from ml_patterns_lab.demand.forecast import forecast_horizon
from ml_patterns_lab.serving import APIModel, ModelDependency, VersionedResponse, create_app


class ForecastRequest(APIModel):
    history: list[Annotated[float, Field(ge=0)]] = Field(min_length=24, max_length=10000)
    last_timestamp: datetime
    horizon: int = Field(default=24, ge=1, le=48)


class ForecastPoint(APIModel):
    timestamp: datetime
    demand_kwh: float = Field(ge=0)


class ForecastResponse(VersionedResponse):
    frequency: Literal["hourly"]
    forecasts: list[ForecastPoint]


app = create_app(
    "demand",
    title="Hourly demand forecast",
    description="Recursive forecasts evaluated at one-hour and 24-hour horizons.",
)


@app.post("/predict", response_model=ForecastResponse)
def predict(payload: ForecastRequest, artifact: ModelDependency) -> dict:
    forecasts = forecast_horizon(
        artifact.bundle, payload.history, payload.last_timestamp, payload.horizon
    )
    return {"frequency": "hourly", "model_version": artifact.run_id, "forecasts": forecasts}
