import importlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECTS = ("churn", "demand", "support", "fraud", "vision", "recommend", "sensors", "mlops")
REQUESTS = {
    "churn": {
        "tenure_months": 12,
        "monthly_spend": 90,
        "support_tickets_90d": 2,
        "plan": "premium",
        "region": "sul",
    },
    "demand": {"history": [100] * 24, "last_timestamp": "2026-01-01T00:00:00", "horizon": 3},
    "support": {"text": "minha fatura mostra uma cobrança duplicada"},
    "fraud": {
        "transaction_id": "txn-demo",
        "amount": 200,
        "transactions_last_hour": 3,
        "merchant_risk_score": 0.4,
        "distance_from_usual_km": 40,
        "is_new_device": True,
        "is_cross_border": False,
    },
    "vision": {"pixels": [0.2] * 576},
    "recommend": {"user_id": "user-001", "limit": 5},
    "sensors": {
        "readings": [{"temperature_c": 52, "vibration_mm_s": 2.6, "pressure_kpa": 99}] * 24
    },
    "mlops": {
        "hours_since_service": 900,
        "temperature_c": 70,
        "vibration_mm_s": 3.5,
        "load_percent": 60,
        "machine_type": "pump",
    },
}


@pytest.mark.parametrize("project", PROJECTS)
def test_real_training_health_and_inference(project, trained_artifacts):
    api = importlib.import_module(f"ml_patterns_lab.{project}.api")
    with TestClient(api.app) as client:
        assert (
            client.get("/health").json()["model_version"] == trained_artifacts[project][1]["run_id"]
        )
        endpoint = "/recommend" if project == "recommend" else "/predict"
        response = client.post(endpoint, json=REQUESTS[project])
        assert response.status_code == 200, response.text
        assert response.json()["model_version"] == trained_artifacts[project][1]["run_id"]
        assert client.get("/openapi.json").status_code == 200
        assert client.post(endpoint, json={**REQUESTS[project], "unexpected": 1}).status_code == 422


@pytest.mark.parametrize("project", ["vision", "demand", "sensors"])
def test_nonfinite_input_returns_validation_error(project, trained_artifacts):
    api = importlib.import_module(f"ml_patterns_lab.{project}.api")
    payload = json.dumps(REQUESTS[project])
    if project == "vision":
        payload = payload.replace("0.2", "NaN", 1)
    elif project == "demand":
        payload = payload.replace("100", "NaN", 1)
    else:
        payload = payload.replace("52", "NaN", 1)
    with TestClient(api.app) as client:
        assert (
            client.post(
                "/predict", content=payload, headers={"Content-Type": "application/json"}
            ).status_code
            == 422
        )


def test_recommendation_batch_unknown_users_and_empty_search(trained_artifacts):
    api = importlib.import_module("ml_patterns_lab.recommend.api")
    with TestClient(api.app) as client:
        assert client.post("/recommend", json={"user_id": "missing"}).status_code == 404
        result = client.post("/recommend/batch", json={"user_ids": ["user-001", "missing"]}).json()
        assert result["errors_by_user_id"] == {"missing": "unknown_user_id"}
        assert set(result["predictions_by_user_id"]) == {"user-001"}
        assert (
            client.post("/recommend/batch", json={"user_ids": ["user-001", "user-001"]}).status_code
            == 422
        )
        assert client.post("/search", json={"query": "xyzzy"}).json()["results"] == []
        assert client.post("/search", json={"query": "   "}).status_code == 422
        assert client.post("/search", json={"query": "programação"}).json()["results"]


def test_sensors_require_enough_readings(trained_artifacts):
    api = importlib.import_module("ml_patterns_lab.sensors.api")
    with TestClient(api.app) as client:
        assert (
            client.post(
                "/predict", json={"readings": REQUESTS["sensors"]["readings"][:23]}
            ).status_code
            == 422
        )


def test_api_fails_startup_when_model_is_missing(tmp_path, monkeypatch):
    api = importlib.import_module("ml_patterns_lab.churn.api")
    monkeypatch.setenv("MLDP_ARTIFACTS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="train"), TestClient(api.app):
        pass


def test_trained_probabilities_and_metrics_are_finite(trained_artifacts):
    for project in ("churn", "fraud", "mlops"):
        bundle, metadata = trained_artifacts[project]
        model = bundle if project == "churn" else bundle["model"]
        module = importlib.import_module(f"ml_patterns_lab.{project}.data")
        data = module.make_transactions(200) if project == "fraud" else module.make_dataset(200)
        probabilities = model.predict_proba(data[module.FEATURES])
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        assert metadata["split_rows"]["train"] > 0


@pytest.mark.parametrize(
    ("project", "field", "value"),
    [
        ("churn", "region", "unknown"),
        ("fraud", "merchant_risk_score", 2),
        ("vision", "pixels", [0.2] * 575),
        ("mlops", "machine_type", "unknown"),
        ("support", "text", "   "),
        ("demand", "history", [-1] * 24),
    ],
)
def test_api_rejects_domain_boundary_violations(project, field, value, trained_artifacts):
    api = importlib.import_module(f"ml_patterns_lab.{project}.api")
    with TestClient(api.app) as client:
        assert client.post("/predict", json={**REQUESTS[project], field: value}).status_code == 422


def test_support_routes_known_domain_and_abstains_on_unknown_vocabulary(trained_artifacts):
    api = importlib.import_module("ml_patterns_lab.support.api")
    assert trained_artifacts["support"][1]["metrics"]["cascade_coverage"] > 0.5
    with TestClient(api.app) as client:
        known = client.post("/predict", json=REQUESTS["support"]).json()
        assert known["intent"] == "cobranca"
        unknown = client.post("/predict", json={"text": "xyzzy qwerty asdfgh"}).json()
        assert unknown["human_review_required"]
