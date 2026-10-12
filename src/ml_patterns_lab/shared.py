"""Small helpers shared by the independent project examples."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


def artifact_root() -> Path:
    """Return the artifact directory, configurable for local or container use."""
    return Path(os.environ.get("MLDP_ARTIFACTS_DIR", "artifacts"))


def save_bundle(project: str, bundle: Any, metadata: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Write an immutable model version and atomically update its current pointer."""
    project_root = artifact_root() / project
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    version_root = project_root / run_id
    version_root.mkdir(parents=True, exist_ok=False)

    model_tmp = version_root / "model.joblib.tmp"
    joblib.dump(bundle, model_tmp)
    model_tmp.replace(version_root / "model.joblib")

    saved_metadata = {
        **metadata,
        "project": project,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_tmp = version_root / "metadata.json.tmp"
    metadata_tmp.write_text(
        json.dumps(saved_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata_tmp.replace(version_root / "metadata.json")

    project_root.mkdir(parents=True, exist_ok=True)
    pointer_tmp = project_root / "latest.json.tmp"
    pointer_tmp.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    pointer_tmp.replace(project_root / "latest.json")
    return version_root, saved_metadata


def load_bundle(project: str) -> tuple[Any, dict[str, Any]]:
    """Load the model and metadata selected by the project's current pointer."""
    project_root = artifact_root() / project
    pointer_path = project_root / "latest.json"
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"No model version for {project!r}; train it before starting the API."
        )

    run_id = json.loads(pointer_path.read_text(encoding="utf-8"))["run_id"]
    version_root = project_root / run_id
    bundle = joblib.load(version_root / "model.joblib")
    metadata = json.loads((version_root / "metadata.json").read_text(encoding="utf-8"))
    return bundle, metadata
