"""HTTP contracts for keyed search and recommendation results."""

from typing import Annotated

from fastapi import HTTPException
from pydantic import Field, field_validator

from ml_patterns_lab.recommend.engine import recommend_items, search_items
from ml_patterns_lab.serving import (
    APIModel,
    ModelDependency,
    Probability,
    VersionedResponse,
    create_app,
)

UserID = Annotated[str, Field(min_length=1, max_length=80)]


class RecommendationRequest(APIModel):
    user_id: UserID
    limit: int = Field(default=10, ge=1, le=50)


class BatchRecommendationRequest(APIModel):
    user_ids: list[UserID] = Field(min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("user_ids")
    @classmethod
    def unique_users(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("user_ids must be unique.")
        return values


class SearchRequest(APIModel):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=10, ge=1, le=50)


class SearchItem(APIModel):
    item_id: str
    title: str
    category: str
    score: Probability


class FeatureCrosses(APIModel):
    topic_x_quality: Probability
    language_x_topic: Probability


class RecommendationItem(SearchItem):
    user_id: str
    feature_crosses: FeatureCrosses


class RecommendationResponse(VersionedResponse):
    user_id: str
    recommendations: list[RecommendationItem]


class BatchResponse(VersionedResponse):
    predictions_by_user_id: dict[str, list[RecommendationItem]]
    errors_by_user_id: dict[str, str]


class SearchResponse(VersionedResponse):
    query: str
    results: list[SearchItem]


app = create_app(
    "recommend",
    title="Content search and recommendations",
    description="Synthetic catalog with hybrid ranking and explicit batch errors.",
)


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest, artifact: ModelDependency) -> dict:
    try:
        results = recommend_items(artifact.bundle, payload.user_id, payload.limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown user_id.") from error
    return {
        "user_id": payload.user_id,
        "recommendations": results,
        "model_version": artifact.run_id,
    }


@app.post("/recommend/batch", response_model=BatchResponse)
def recommend_batch(payload: BatchRecommendationRequest, artifact: ModelDependency) -> dict:
    results, errors = {}, {}
    for user_id in payload.user_ids:
        try:
            results[user_id] = recommend_items(artifact.bundle, user_id, payload.limit)
        except KeyError:
            errors[user_id] = "unknown_user_id"
    return {
        "predictions_by_user_id": results,
        "errors_by_user_id": errors,
        "model_version": artifact.run_id,
    }


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, artifact: ModelDependency) -> dict:
    return {
        "query": payload.query,
        "results": search_items(artifact.bundle, payload.query, payload.limit),
        "model_version": artifact.run_id,
    }
