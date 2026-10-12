"""Serving endpoint for the currently registered maintenance-risk model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from ml_patterns_lab.shared import load_bundle


class MaintenanceRequest(BaseModel):
    hours_since_service: float = Field(ge=0, le=20000)
    temperature_c: float = Field(ge=-10, le=150)
    vibration_mm_s: float = Field(ge=0, le=30)
    load_percent: float = Field(ge=0, le=100)
    machine_type: Literal["pump", "compressor", "conveyor"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle, metadata = load_bundle("mlops")
    app.state.bundle = bundle
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Maintenance-risk model service",
    description="Synthetic equipment example produced by the reproducible workflow.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.metadata["run_id"]}


@app.post("/predict")
def predict(payload: MaintenanceRequest, request: Request) -> dict:
    metadata = request.app.state.metadata
    bundle = request.app.state.bundle
    features = pd.DataFrame([payload.model_dump()])
    probability = float(bundle["model"].predict_proba(features)[0, 1])
    return {
        "maintenance_risk": round(probability, 4),
        "threshold": round(float(bundle["threshold"]), 4),
        "recommendation": (
            "schedule_maintenance_review"
            if probability >= float(bundle["threshold"])
            else "continue_monitoring"
        ),
        "model_version": metadata["run_id"],
    }
