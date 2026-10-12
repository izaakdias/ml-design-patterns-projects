"""Regression tests for publication, concurrency, integrity and rollback."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from ml_patterns_lab.shared import artifact_root, load_bundle, promote_bundle, save_bundle


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MLDP_ARTIFACTS_DIR", str(tmp_path))


def test_roundtrip_documents_and_rollback():
    first_path, first = save_bundle(
        "demo", {"value": 1}, {}, documents={"workflow.json": {"done": True}}
    )
    _, second = save_bundle("demo", {"value": 2}, {})
    assert load_bundle("demo")[0] == {"value": 2}
    assert "workflow.json" in json.loads((first_path / "manifest.json").read_text())
    promote_bundle("demo", first["run_id"])
    assert load_bundle("demo")[0] == {"value": 1}
    assert load_bundle("demo", second["run_id"])[0] == {"value": 2}


def test_checkpoint_does_not_replace_complete_model():
    _, original = save_bundle("demo", {"complete": True}, {})
    _, checkpoint = save_bundle("demo", {"complete": False}, {}, promote=False)
    assert load_bundle("demo")[1]["run_id"] == original["run_id"]
    assert load_bundle("demo", checkpoint["run_id"])[0] == {"complete": False}


def test_failed_write_preserves_latest_and_cleans_staging():
    _, original = save_bundle("demo", {"complete": True}, {})
    with pytest.raises(ValueError):
        save_bundle("demo", {}, {"bad": float("nan")})
    assert load_bundle("demo")[1]["run_id"] == original["run_id"]
    assert not list((artifact_root() / "demo").glob(".staging-*"))


def test_concurrent_writers_publish_complete_unique_versions():
    def publish(index):
        return save_bundle("demo", {"index": index}, {})

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(publish, range(12)))
    run_ids = {metadata["run_id"] for _, metadata in results}
    assert len(run_ids) == 12
    assert load_bundle("demo")[1]["run_id"] in run_ids
    for path, metadata in results:
        assert load_bundle("demo", metadata["run_id"])[0]["index"] in range(12)
        assert sorted(item.name for item in path.iterdir()) == [
            "manifest.json",
            "metadata.json",
            "model.joblib",
        ]


@pytest.mark.parametrize("file", ["model.joblib", "metadata.json", "workflow.json"])
def test_corruption_is_detected_before_deserialization(file):
    path, _ = save_bundle("demo", {}, {}, documents={"workflow.json": {}})
    (path / file).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        load_bundle("demo")


@pytest.mark.parametrize("name", ["../other", "/tmp/other", "..", "", "a/b"])
def test_names_cannot_escape_registry(name):
    with pytest.raises(ValueError):
        save_bundle(name, {}, {})


def test_missing_model_has_actionable_error():
    with pytest.raises(FileNotFoundError, match="train"):
        load_bundle("missing")


def test_invalid_document_cannot_change_latest():
    _, original = save_bundle("demo", {}, {})
    with pytest.raises(ValueError):
        save_bundle("demo", {}, {}, documents={"metadata.json": {}})
    assert load_bundle("demo")[1]["run_id"] == original["run_id"]


def test_failed_model_serialization_preserves_current_version():
    _, original = save_bundle("demo", {}, {})
    with pytest.raises(TypeError):
        save_bundle("demo", Lock(), {})
    assert load_bundle("demo")[1]["run_id"] == original["run_id"]


def test_runtime_mismatch_has_actionable_error(monkeypatch):
    save_bundle("demo", {}, {})
    monkeypatch.setattr("ml_patterns_lab.shared.version", lambda _package: "different")
    with pytest.raises(ValueError, match="retrain"):
        load_bundle("demo")


def test_http_process_keeps_its_pinned_version_after_promotion(monkeypatch):
    from fastapi.testclient import TestClient

    from ml_patterns_lab.serving import create_app

    _, original = save_bundle("demo", {"value": 1}, {})
    _, replacement = save_bundle("demo", {"value": 2}, {})
    monkeypatch.setenv("MLDP_MODEL_VERSION", original["run_id"])
    app = create_app("demo", title="Demo", description="Version consistency")
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == original["run_id"]
        promote_bundle("demo", replacement["run_id"])
        assert client.get("/health").json()["model_version"] == original["run_id"]
