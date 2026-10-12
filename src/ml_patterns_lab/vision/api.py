"""HTTP contracts for flattened 24-by-24 grayscale images."""

from typing import Annotated, Literal

from pydantic import Field

from ml_patterns_lab.serving import (
    APIModel,
    ModelDependency,
    Probability,
    VersionedResponse,
    create_app,
)
from ml_patterns_lab.vision.cascade import route_image
from ml_patterns_lab.vision.data import IMAGE_SIZE


class ImageRequest(APIModel):
    pixels: Annotated[list[Probability], Field(min_length=IMAGE_SIZE, max_length=IMAGE_SIZE)]


class ImageResponse(VersionedResponse):
    inspection: Literal["defect", "clear", "manual_review"]
    confidence: Probability
    stage: Literal["fast", "expert", "review"]
    manual_review_required: bool


app = create_app(
    "vision",
    title="Visual quality inspection",
    description="Synthetic domain adaptation, input quantization and confidence-based review.",
)


@app.post("/predict", response_model=ImageResponse)
def predict(payload: ImageRequest, artifact: ModelDependency) -> dict:
    result = route_image(
        artifact.bundle,
        payload.pixels,
        artifact.metadata["fast_confidence_threshold"],
        artifact.metadata["expert_review_threshold"],
    )
    return {**result, "model_version": artifact.run_id}
