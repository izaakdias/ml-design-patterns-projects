"""Keyed search and recommendation endpoints over one shared feature snapshot."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ml_patterns_lab.recommend.engine import recommend_items, search_items
from ml_patterns_lab.shared import load_bundle


class RecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    limit: int = Field(default=10, ge=1, le=50)


class BatchRecommendationRequest(BaseModel):
    user_ids: list[str] = Field(min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=50)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=10, ge=1, le=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle, metadata = load_bundle("recommend")
    app.state.store = bundle
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Content search and recommendations",
    description="Synthetic catalog with TF-IDF retrieval, interaction embeddings, and keyed results.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.metadata["run_id"]}


@app.post("/recommend")
def recommend(payload: RecommendationRequest, request: Request) -> dict:
    try:
        results = recommend_items(request.app.state.store, payload.user_id, payload.limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "user_id": payload.user_id,
        "recommendations": results,
        "model_version": request.app.state.metadata["run_id"],
    }


@app.post("/recommend/batch")
def recommend_batch(payload: BatchRecommendationRequest, request: Request) -> dict:
    keyed_results = {}
    for user_id in payload.user_ids:
        try:
            keyed_results[user_id] = recommend_items(
                request.app.state.store,
                user_id,
                payload.limit,
            )
        except KeyError:
            keyed_results[user_id] = []
    return {
        "predictions_by_user_id": keyed_results,
        "model_version": request.app.state.metadata["run_id"],
    }


@app.post("/search")
def search(payload: SearchRequest, request: Request) -> dict:
    try:
        results = search_items(request.app.state.store, payload.query, payload.limit)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "query": payload.query,
        "results": results,
        "model_version": request.app.state.metadata["run_id"],
    }
