"""Stateless HTTP endpoint for Portuguese-language support-ticket triage."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from ml_patterns_lab.shared import load_bundle
from ml_patterns_lab.support.cascade import route_ticket


class TicketRequest(BaseModel):
    text: str = Field(min_length=8, max_length=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle, metadata = load_bundle("support")
    app.state.bundle = bundle
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Support-ticket triage",
    description="Educational Portuguese-language triage example using generated text.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.metadata["run_id"]}


@app.post("/predict")
def predict(payload: TicketRequest, request: Request) -> dict:
    metadata = request.app.state.metadata
    result = route_ticket(
        request.app.state.bundle["fast"],
        request.app.state.bundle["expert"],
        payload.text,
        float(metadata["fast_threshold"]),
        float(metadata["human_review_threshold"]),
    )
    return {**result, "model_version": metadata["run_id"]}
