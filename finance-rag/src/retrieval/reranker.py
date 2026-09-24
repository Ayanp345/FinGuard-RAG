"""
Cross-encoder reranking.

Bi-encoders (used for dense retrieval) embed the query and each chunk
independently, so they can never model query-chunk *interaction* — they're
fast but comparatively imprecise. A cross-encoder scores (query, chunk)
pairs jointly and is far more accurate at judging "does this chunk actually
answer this question", at the cost of being too slow to run over the whole
corpus. Using it only on the ~15 RRF-fused candidates gets the best of
both: bi-encoder recall, cross-encoder precision.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates[:top_k]
