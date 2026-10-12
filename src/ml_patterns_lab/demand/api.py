"""Stateless HTTP API for recursively forecast hourly demand."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from ml_patterns_lab.demand.forecast import forecast_horizon
from ml_patterns_lab.shared import load_bundle


class ForecastRequest(BaseModel):
    history: list[float] = Field(min_length=24, max_length=10000)
    last_timestamp: datetime
    horizon: int = Field(default=24, ge=1, le=48)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle, metadata = load_bundle("demand")
    app.state.model = bundle
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Hourly demand forecast",
    description="Educational forecasting example using generated hourly data.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.metadata["run_id"]}


@app.post("/predict")
def predict(payload: ForecastRequest, request: Request) -> dict:
    forecasts = forecast_horizon(
        request.app.state.model,
        payload.history,
        payload.last_timestamp,
        payload.horizon,
    )
    return {
        "frequency": "hourly",
        "model_version": request.app.state.metadata["run_id"],
        "forecasts": forecasts,
    }
