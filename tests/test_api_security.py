from fastapi.testclient import TestClient

from llm_eval.api import app


def test_remote_token_mode_requires_bearer(monkeypatch):
    monkeypatch.setenv("LLM_EVAL_API_TOKEN", "test-secret")
    client = TestClient(app)
    payload = {
        "experiment": "auth-test",
        "items": [
            {
                "id": "case-1",
                "prompt": "What is RAG?",
                "output": "retrieval augmented generation",
                "expected_terms": ["retrieval"],
            }
        ],
    }

    denied = client.post("/v1/evaluate", json=payload)
    assert denied.status_code == 401

    allowed = client.post(
        "/v1/evaluate",
        json=payload,
        headers={"Authorization": "Bearer test-secret"},
    )
    assert allowed.status_code == 200
