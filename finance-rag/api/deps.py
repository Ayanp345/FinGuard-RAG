from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from fastapi import Header, HTTPException, Request, status

from src.config import Settings, settings
from src.generation.llm_client import build_llm_client
from src.hallucination_checker import HallucinationChecker
from src.pipeline import RAGPipeline
from src.query_classifier import QueryClassifier
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)


class AppState:
    """Holds every heavyweight object that should be loaded exactly once at
    process startup, not per-request. Attached to `app.state` in main.py."""

    pipeline: RAGPipeline | None = None
    redis_client = None

    def load(self, cfg: Settings) -> None:
        index_dir = Path(cfg.index_dir)
        dense = DenseRetriever.load(index_dir, device=cfg.device)
        sparse = SparseRetriever.load(index_dir)
        hybrid = HybridRetriever(dense, sparse, rrf_k=cfg.rrf_k)
        reranker = Reranker(cfg.reranker_model, cfg.device)
        classifier = QueryClassifier(dense.encoder)
        hallucination_checker = HallucinationChecker(cfg.nli_model, cfg.device)
        llm_client = build_llm_client(cfg)
        self.pipeline = RAGPipeline(hybrid, reranker, classifier, llm_client, hallucination_checker, cfg)

        if cfg.redis_url:
            import redis.asyncio as aioredis

            self.redis_client = aioredis.from_url(cfg.redis_url, decode_responses=True)
            logger.info("Response caching enabled via %s", cfg.redis_url)

    async def get_cached(self, query: str) -> dict | None:
        if self.redis_client is None:
            return None
        key = "ragcache:" + hashlib.sha256(query.strip().lower().encode()).hexdigest()
        raw = await self.redis_client.get(key)
        return json.loads(raw) if raw else None

    async def set_cached(self, query: str, payload: dict, ttl_seconds: int = 3600) -> None:
        if self.redis_client is None:
            return
        key = "ragcache:" + hashlib.sha256(query.strip().lower().encode()).hexdigest()
        await self.redis_client.set(key, json.dumps(payload), ex=ttl_seconds)


app_state = AppState()


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = request.app.state.app_state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized.")
    return pipeline


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key is None:
        return  # auth disabled when no API key is configured
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key.")
