"""Integration fixtures train each real project once in an isolated registry."""

import importlib

import pytest

from ml_patterns_lab.shared import load_bundle

PROJECTS = ("churn", "demand", "support", "fraud", "vision", "recommend", "sensors", "mlops")


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLDP_ARTIFACTS_DIR", str(tmp_path_factory.mktemp("models")))
        patch.delenv("MLDP_MODEL_VERSION", raising=False)
        artifacts = {}
        for project in PROJECTS:
            module, function = (
                ("mlops.pipeline", "run_pipeline")
                if project == "mlops"
                else (f"{project}.train", "train")
            )
            getattr(importlib.import_module(f"ml_patterns_lab.{module}"), function)()
            artifacts[project] = load_bundle(project)
        yield artifacts
