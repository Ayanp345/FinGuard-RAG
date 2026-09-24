"""
API-level tests.

`AppState.load` normally downloads/loads embedding, reranker, NLI and LLM
models — far too heavy for a unit test. We patch it to install a stub
pipeline instead, so these tests exercise routing, auth, caching and
response shaping without needing any model weights or network access.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models import QueryType, RAGAnswer


class _StubPipeline:
    async def answer(self, query: str) -> RAGAnswer:
        return RAGAnswer(
            query=query,
            query_type=QueryType.FACTUAL,
            answer="Stubbed answer for testing.",
            citations=[],
            faithfulness_score=1.0,
            unsupported_claim_count=0,
            retrieved_chunk_ids=["c1"],
            latency_ms=12.3,
        )


def _make_client() -> TestClient:
    with patch("api.deps.AppState.load") as mock_load:
        def fake_load(self, cfg):
            self.pipeline = _StubPipeline()

        mock_load.side_effect = fake_load
        from api.main import app  # imported here so the patch is active during lifespan startup

        return TestClient(app)


@pytest.fixture
def client():
    with _make_client() as c:
        yield c


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True


def test_query_endpoint_returns_stubbed_answer(client: TestClient):
    resp = client.post("/query", json={"query": "What was TCS revenue in FY2024?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Stubbed answer for testing."
    assert body["query_type"] == "factual"
    assert body["faithfulness_score"] == 1.0
    assert body["cached"] is False


def test_query_endpoint_rejects_too_short_query(client: TestClient):
    resp = client.post("/query", json={"query": "hi"})
    assert resp.status_code == 422  # fails min_length validation


def test_query_endpoint_requires_api_key_when_configured(client: TestClient, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "api_key", "secret123")
    resp = client.post("/query", json={"query": "What was TCS revenue in FY2024?"})
    assert resp.status_code == 401

    resp = client.post(
        "/query",
        json={"query": "What was TCS revenue in FY2024?"},
        headers={"X-API-Key": "secret123"},
    )
    assert resp.status_code == 200
