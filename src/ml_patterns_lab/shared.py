"""Local model registry: immutable, verified versions and atomic promotion."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib

SCHEMA_VERSION = 1
_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")


def artifact_root() -> Path:
    return Path(os.environ.get("MLDP_ARTIFACTS_DIR", "artifacts")).resolve()


def _safe_name(value: str) -> str:
    if not _NAME.fullmatch(value):
        raise ValueError(f"Invalid artifact name: {value!r}")
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _promote(project_root: Path, run_id: str) -> None:
    # Unique temporary pointers prevent concurrent writers sharing a staging path.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=project_root, delete=False, encoding="utf-8"
    ) as handle:
        pointer = Path(handle.name)
        try:
            json.dump({"run_id": run_id}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            pointer.unlink(missing_ok=True)
            raise
    try:
        pointer.replace(project_root / "latest.json")
    finally:
        pointer.unlink(missing_ok=True)


def save_bundle(
    project: str,
    bundle: Any,
    metadata: dict[str, Any],
    *,
    promote: bool = True,
    documents: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish the entire version before changing latest; checkpoints opt out."""
    project_root = artifact_root() / _safe_name(project)
    project_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}-{uuid4().hex[:12]}"
    version_root = project_root / run_id
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=project_root))
    saved_metadata = {
        **metadata,
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ("numpy", "pandas", "scikit-learn", "joblib")
            },
        },
    }
    try:
        joblib.dump(bundle, staging / "model.joblib", compress=3)
        _write_json(staging / "metadata.json", saved_metadata)
        for name, data in (documents or {}).items():
            _safe_name(name)
            if not name.endswith(".json") or name in {"metadata.json", "manifest.json"}:
                raise ValueError(f"Reserved or unsupported document: {name!r}")
            _write_json(staging / name, data)
        _write_json(
            staging / "manifest.json",
            {path.name: _checksum(path) for path in sorted(staging.iterdir())},
        )
        staging.replace(version_root)
        if promote:
            _promote(project_root, run_id)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return version_root, saved_metadata


def load_bundle(project: str, run_id: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Verify integrity before deserializing trusted local joblib files."""
    project_root = artifact_root() / _safe_name(project)
    if run_id is None:
        pointer = project_root / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError(
                f"No model version for {project!r}; train it before starting the API."
            )
        run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    version_root = project_root / _safe_name(run_id)
    manifest = json.loads((version_root / "manifest.json").read_text(encoding="utf-8"))
    if not {"model.joblib", "metadata.json"}.issubset(manifest):
        raise ValueError("Incomplete artifact manifest.")
    for name, expected in manifest.items():
        path = version_root / _safe_name(name)
        if not path.is_file() or _checksum(path) != expected:
            raise ValueError(f"Artifact integrity check failed: {name}")
    metadata = json.loads((version_root / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("project") != project
        or metadata.get("run_id") != run_id
    ):
        raise ValueError("Artifact metadata does not match the selected version or schema.")
    if metadata.get("runtime", {}).get("scikit-learn") != version("scikit-learn"):
        raise ValueError(
            "scikit-learn version differs from training; restore that environment or retrain."
        )
    return joblib.load(version_root / "model.joblib"), metadata


def promote_bundle(project: str, run_id: str) -> None:
    """Promote or roll back to an existing, verified version."""
    load_bundle(project, run_id)
    _promote(artifact_root() / _safe_name(project), _safe_name(run_id))
