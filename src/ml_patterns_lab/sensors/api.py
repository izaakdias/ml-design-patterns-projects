"""HTTP contracts for recent sensor windows."""

from pydantic import Field

from ml_patterns_lab.sensors.features import WINDOW_SIZE
from ml_patterns_lab.sensors.inference import score_window
from ml_patterns_lab.serving import APIModel, ModelDependency, VersionedResponse, create_app


class SensorReading(APIModel):
    temperature_c: float = Field(ge=-100, le=300)
    vibration_mm_s: float = Field(ge=0, le=100)
    pressure_kpa: float = Field(gt=0, le=10000)


class SensorWindowRequest(APIModel):
    readings: list[SensorReading] = Field(min_length=WINDOW_SIZE, max_length=10000)


class Deviation(APIModel):
    feature: str
    scaled_deviation: float = Field(ge=0)


class SensorResponse(VersionedResponse):
    anomaly_score: float
    anomaly_flag: bool
    threshold: float
    top_window_deviations: list[Deviation]


app = create_app(
    "sensors",
    title="Sensor anomaly monitor",
    description="Window-based anomaly scores with deviations from the normal reference.",
)


@app.post("/predict", response_model=SensorResponse)
def predict(payload: SensorWindowRequest, artifact: ModelDependency) -> dict:
    result = score_window(
        artifact.bundle, artifact.metadata, [item.model_dump() for item in payload.readings]
    )
    return {**result, "model_version": artifact.run_id}
