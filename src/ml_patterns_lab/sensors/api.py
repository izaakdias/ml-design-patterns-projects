"""Windowed sensor scoring endpoint for an explicit batch of recent readings."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ml_patterns_lab.sensors.data import SENSOR_COLUMNS
from ml_patterns_lab.sensors.features import WINDOW_SIZE
from ml_patterns_lab.sensors.inference import score_window
from ml_patterns_lab.shared import load_bundle


class SensorReading(BaseModel):
    temperature_c: float
    vibration_mm_s: float
    pressure_kpa: float


class SensorWindowRequest(BaseModel):
    readings: list[SensorReading] = Field(min_length=WINDOW_SIZE, max_length=10000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle, metadata = load_bundle("sensors")
    app.state.bundle = bundle
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Sensor anomaly monitor",
    description="Window-based anomaly score and feature-deviation hints on synthetic telemetry.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.metadata["run_id"]}


@app.post("/predict")
def predict(payload: SensorWindowRequest, request: Request) -> dict:
    metadata = request.app.state.metadata
    readings = [item.model_dump(include=set(SENSOR_COLUMNS)) for item in payload.readings]
    try:
        result = score_window(request.app.state.bundle, metadata, readings)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {**result, "model_version": metadata["run_id"]}
