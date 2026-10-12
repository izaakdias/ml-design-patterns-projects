"""Keyed transaction-risk API with review bands instead of automatic blocking."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from ml_patterns_lab.shared import load_bundle


class TransactionRequest(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=80)
    amount: float = Field(gt=0, le=100000)
    transactions_last_hour: int = Field(ge=0, le=1000)
    merchant_risk_score: float = Field(ge=0, le=1)
    distance_from_usual_km: float = Field(ge=0, le=50000)
    is_new_device: bool
    is_cross_border: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle, metadata = load_bundle("fraud")
    app.state.bundle = bundle
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Transaction-risk triage",
    description="Educational rare-event scoring on synthetic transactions; every alert requires review.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.metadata["run_id"]}


@app.post("/predict")
def predict(payload: TransactionRequest, request: Request) -> dict:
    bundle = request.app.state.bundle
    metadata = request.app.state.metadata
    features = pd.DataFrame(
        [
            {
                name: getattr(payload, name)
                for name in metadata["features"]
            }
        ]
    )
    score = float(bundle["model"].predict_proba(features)[0, 1])
    if score >= float(bundle["urgent_threshold"]):
        action = "urgent_human_review"
    elif score >= float(bundle["review_threshold"]):
        action = "human_review"
    else:
        action = "standard_queue"
    return {
        "transaction_id": payload.transaction_id,
        "fraud_score": round(score, 4),
        "action": action,
        "model_version": metadata["run_id"],
    }
