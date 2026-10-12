"""HTTP lifecycle and contracts shared by the independent model services."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ml_patterns_lab.shared import load_bundle


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class HealthResponse(APIModel):
    status: str
    model_version: str


class VersionedResponse(APIModel):
    model_version: str


Probability = Annotated[float, Field(ge=0, le=1)]


@dataclass(frozen=True)
class Artifact:
    """Serialization boundary; each project owns its model's internal shape."""

    bundle: Any
    metadata: dict[str, Any]

    @property
    def run_id(self) -> str:
        return self.metadata["run_id"]


def get_artifact(request: Request) -> Artifact:
    return request.app.state.artifact


ModelDependency = Annotated[Artifact, Depends(get_artifact)]


def create_app(project: str, *, title: str, description: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bundle, metadata = load_bundle(project, os.environ.get("MLDP_MODEL_VERSION") or None)
        app.state.artifact = Artifact(bundle, metadata)
        yield
        del app.state.artifact

    app = FastAPI(title=title, description=description, version="0.2.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        # Raw NaN/Infinity inputs cannot be echoed through a strict JSON encoder.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: item[key] for key in ("loc", "msg", "type")} for item in error.errors()
                ]
            },
        )

    @app.get("/health", response_model=HealthResponse)
    def health(artifact: ModelDependency) -> HealthResponse:
        return HealthResponse(status="ok", model_version=artifact.run_id)

    return app
