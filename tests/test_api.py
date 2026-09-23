"""API 层单测（FastAPI TestClient，离线）。作者：晨星"""
import pytest
from fastapi.testclient import TestClient

from starlight.app import create_app
from starlight.config import Settings

DOC = "星环引擎采用氦-3 燃料循环，额定功率 42 兆瓦。" * 50  # 约 1100 字符，显式超过 800 分块阈值


@pytest.fixture
def client():
    return TestClient(create_app(Settings()))


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestDocuments:
    def test_ingest_success(self, client):
        resp = client.post("/api/v1/documents", json={"title": "白皮书", "content": DOC})
        assert resp.status_code == 201
        assert resp.json()["chunks"] > 1

    def test_ingest_empty_content_422(self, client):
        resp = client.post("/api/v1/documents", json={"title": "空", "content": ""})
        assert resp.status_code == 422

    def test_list_and_delete(self, client):
        resp = client.post(
            "/api/v1/documents", json={"title": "白皮书", "content": DOC, "doc_id": "d-x"}
        )
        assert resp.status_code == 201
        listing = client.get("/api/v1/documents")
        assert any(d["doc_id"] == "d-x" for d in listing.json())
        assert client.delete("/api/v1/documents/d-x").status_code == 204
        remaining = client.get("/api/v1/documents").json()
        assert not any(d["doc_id"] == "d-x" for d in remaining)

    def test_delete_missing_404(self, client):
        assert client.delete("/api/v1/documents/nope").status_code == 404


class TestQuery:
    def test_query_success_flow(self, client):
        client.post("/api/v1/documents", json={"title": "白皮书", "content": DOC, "doc_id": "d-q"})
        resp = client.post("/api/v1/query", json={"question": "星环引擎的额定功率"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"]
        assert body["sources"][0]["doc_id"] == "d-q"

    def test_query_empty_question_422(self, client):
        assert client.post("/api/v1/query", json={"question": ""}).status_code == 422


class TestEvaluate:
    def test_evaluate_endpoint_fresh_pipeline(self, client):
        payload = {
            "corpus": [{"doc_id": "ev-1", "title": "白皮书", "content": DOC}],
            "cases": [{"question": "星环引擎的燃料", "expect_doc_id": "ev-1"}],
            "k": 3,
        }
        resp = client.post("/api/v1/evaluate", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["cases"] == 1
        assert body["recall_at_k"] == 1.0
        # 评估在全新管道上运行，不应污染主服务状态
        assert client.get("/health").json()["chunks"] == 0
