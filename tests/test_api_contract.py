import json
from pathlib import Path

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from llm_eval.api import app

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "contracts" / "makma.v1.schema.json").read_text(
        encoding="utf-8"
    )
)
VALIDATOR = Draft202012Validator(SCHEMA)
AUTH = {"Authorization": "Bearer test-secret"}
CITATIONS = {"cited_ids": ["d1"], "evidence_ids": ["d1", "d2"]}


def test_contract_routes_require_bearer_token_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_EVAL_API_TOKEN", "test-secret")
    client = TestClient(app)
    assert client.get("/v1/capabilities").status_code == 401
    assert client.post("/v1/ops/validate_citations", json=CITATIONS).status_code == 401
    assert client.get("/v1/capabilities", headers=AUTH).status_code == 200


def test_contract_routes_reject_remote_clients_without_token(monkeypatch):
    monkeypatch.delenv("LLM_EVAL_API_TOKEN", raising=False)
    client = TestClient(app, base_url="http://eval.example.com")
    assert client.get("/v1/capabilities").status_code == 401


def test_ops_route_returns_envelope_with_correlation_id(monkeypatch):
    monkeypatch.setenv("LLM_EVAL_API_TOKEN", "test-secret")
    response = TestClient(app).post(
        "/v1/ops/validate_citations",
        json=CITATIONS,
        headers={**AUTH, "X-Correlation-ID": "corr-42"},
    )
    assert response.status_code == 200
    body = response.json()
    VALIDATOR.validate(body)
    assert body["status"] == "ok"
    assert body["correlation_id"] == "corr-42"
    assert body["data"]["valid"] is True


def test_ops_route_maps_error_codes_to_http_status(monkeypatch):
    monkeypatch.setenv("LLM_EVAL_API_TOKEN", "test-secret")
    client = TestClient(app)
    unknown = client.post("/v1/ops/delete_everything", json={}, headers=AUTH)
    assert unknown.status_code == 404
    VALIDATOR.validate(unknown.json())
    assert unknown.json()["error"]["code"] == "UNKNOWN_OPERATION"
    invalid = client.post("/v1/ops/validate_citations", json={"cited_ids": 3}, headers=AUTH)
    assert invalid.status_code == 422
    VALIDATOR.validate(invalid.json())
    assert invalid.json()["error"]["code"] == "INVALID_INPUT"


def test_existing_routes_still_work(monkeypatch):
    monkeypatch.delenv("LLM_EVAL_API_TOKEN", raising=False)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
