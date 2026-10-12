from fastapi.testclient import TestClient
import pytest
from factorymind import api, core

@pytest.fixture
def client(monkeypatch, trained, tmp_path):
    monkeypatch.setattr(api, "assets", lambda: trained)
    monkeypatch.setattr(api, "REVIEW_PATH", tmp_path / "reviews.sqlite3")
    return TestClient(api.app)


def test_prediction_pagination_and_validation(client):
    assert client.get("/health").json()["cycles"] == 48
    assert client.get("/cycles?limit=2&offset=2").json()["cycles"] == [3, 4]
    assert client.get("/cycles?limit=-1").status_code == 422
    case = client.get("/predict/1").json()
    assert case["sensor_evidence"]["PS1"]["unit"] == "bar"
    assert case["prediction_latency_ms"] > 0
    assert client.get("/predict/9999").status_code == 404
    assert client.post("/investigate", json={"cycle_id": 0}).status_code == 422


def test_missing_model_returns_unready(monkeypatch):
    def missing():
        raise FileNotFoundError("secret path")
    monkeypatch.setattr(api, "assets", missing)
    client = TestClient(api.app)
    for path in ("/health", "/cycles", "/predict/1"):
        result = client.get(path)
        assert result.status_code == 503
        assert "secret path" not in result.text


def test_investigation_and_fallback(client, monkeypatch):
    monkeypatch.setattr(core, "investigate", lambda case, **kwargs: "Validated investigation")
    result = client.post("/investigate", json={"cycle_id": 1}).json()
    assert result["provider_mode"] == "llm"
    def fail(case, **kwargs):
        raise TimeoutError("secret token should never be displayed")
    monkeypatch.setattr(core, "investigate", fail)
    result = client.post("/investigate", json={"cycle_id": 1}).json()
    assert result["provider_mode"] == "fallback"
    assert "secret token" not in result["investigation"]
    assert "not an AI investigation" in result["investigation"]


def test_review_records_both_decisions(client):
    for decision in ("approve_inspection", "reject_inspection"):
        result = client.post("/reviews", json={"cycle_id": 1, "decision": decision, "notes": "Automated test"})
        assert result.status_code == 201
        assert result.json()["equipment_control"] is False
    assert len(client.get("/reviews/1").json()["reviews"]) == 2
    assert client.post("/reviews", json={"cycle_id": 1, "decision": "start_machine"}).status_code == 422
    assert client.post("/reviews", json={"cycle_id": 9999, "decision": "approve_inspection"}).status_code == 404


def test_raw_trace_validation(client, monkeypatch):
    import numpy as np
    monkeypatch.setattr(api, "read_cycle_samples", lambda *args: np.arange(6000))
    result = client.get("/sensors/1/PS1").json()
    assert len(result["values"]) == 600
    assert result["seconds"][-1] == 59.9
    assert client.get("/sensors/1/profile").status_code == 422


def test_investigation_never_creates_a_review(client, monkeypatch):
    monkeypatch.setattr(core, "investigate", lambda case, **kwargs: "Test-only investigation")
    client.post("/investigate", json={"cycle_id": 1})
    assert client.get("/reviews/1").json()["reviews"] == []


def test_provider_health_independent_of_classifier(client, monkeypatch):
    monkeypatch.setattr(api, "check_connection", lambda config: {"ready": False, "failure": "timeout"})
    assert client.get("/health").status_code == 200
    result = client.get("/provider/health")
    assert result.status_code == 503 and result.json()["failure"] == "timeout"
