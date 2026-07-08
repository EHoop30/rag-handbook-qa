from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_indexed_count(service):
    client = TestClient(create_app(service))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0


def test_chat_endpoint_returns_structured_response(service):
    client = TestClient(create_app(service))
    resp = client.post("/chat", json={"question": "How many sick days do I get?", "k": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"]
    assert body["answer"]
    assert len(body["sources"]) <= 4
    assert {"doc_id", "title", "snippet", "score"} <= body["sources"][0].keys()


def test_chat_rejects_empty_question(service):
    client = TestClient(create_app(service))
    resp = client.post("/chat", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length validation
