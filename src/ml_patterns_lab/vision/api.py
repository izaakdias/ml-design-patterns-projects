"""HTTP inspection service accepting a flattened 24-by-24 grayscale image."""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from ml_patterns_lab.shared import load_bundle
from ml_patterns_lab.vision.cascade import route_image
from ml_patterns_lab.vision.data import IMAGE_SIZE


class ImageRequest(BaseModel):
    pixels: list[float] = Field(min_length=IMAGE_SIZE, max_length=IMAGE_SIZE)

    @field_validator("pixels")
    @classmethod
    def check_pixel_range(cls, values: list[float]) -> list[float]:
        array = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("Each pixel must be a finite value between 0 and 1.")
        return values


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle, metadata = load_bundle("vision")
    app.state.bundle = bundle
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Visual quality inspection",
    description="Synthetic image example with adaptation, quantized inputs, and review fallback.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.metadata["run_id"]}


@app.post("/predict")
def predict(payload: ImageRequest, request: Request) -> dict:
    metadata = request.app.state.metadata
    try:
        result = route_image(
            request.app.state.bundle,
            payload.pixels,
            float(metadata["fast_confidence_threshold"]),
            float(metadata["expert_review_threshold"]),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {**result, "model_version": metadata["run_id"]}
