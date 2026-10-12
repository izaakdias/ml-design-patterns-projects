"""Stateless HTTP inference for customer-success prioritization."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from ml_patterns_lab.shared import load_bundle


class ChurnRequest(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(gt=0, le=1000)
    support_tickets_90d: int = Field(ge=0, le=100)
    plan: Literal["essencial", "conforto", "premium"]
    region: Literal["norte", "sul", "leste", "oeste"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle, metadata = load_bundle("churn")
    app.state.model = bundle
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Customer-success churn prioritization",
    description="Educational example using synthetic data. Review each recommendation with a person.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.metadata["run_id"]}


@app.post("/predict")
def predict(payload: ChurnRequest, request: Request) -> dict:
    model = request.app.state.model
    metadata = request.app.state.metadata
    features = pd.DataFrame([payload.model_dump()])
    probability = float(model.predict_proba(features)[0, 1])
    threshold = float(metadata["threshold"])
    return {
        "churn_probability": round(probability, 4),
        "threshold": round(threshold, 4),
        "recommendation": "review_for_retention" if probability >= threshold else "no_alert",
        "model_version": metadata["run_id"],
    }
