"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Why RRF instead of a weighted sum of raw scores (the "naive hybrid" most
tutorials show): FAISS cosine similarity lives in [-1, 1] and BM25 scores
are unbounded and corpus-dependent, so `alpha * dense_score + (1-alpha) *
bm25_score` is comparing numbers with no shared meaning — the result is
extremely sensitive to `alpha` and re-tuned every time the corpus changes.

RRF sidesteps this entirely by fusing on *rank*, not raw score:

    score(chunk) = sum over retrievers r of  1 / (k + rank_r(chunk))

A chunk that is merely present near the top of both lists consistently
outranks a chunk that is #1 in one list and absent from the other — which
is usually what you want, since it means both lexical and semantic signals
agree. `k` (default 60, per the original Cormack et al. RRF paper) damps
the influence of rank-1 vs rank-2 so the fusion isn't dominated by whichever
retriever happens to be more confident.
"""
from __future__ import annotations

from src.models import Chunk, RetrievedChunk
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 25,
        sparse_top_k: int = 25,
        fused_top_k: int = 15,
    ) -> list[RetrievedChunk]:
        dense_hits = self.dense.search(query, top_k=dense_top_k)
        sparse_hits = self.sparse.search(query, top_k=sparse_top_k)

        dense_rank = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(dense_hits, start=1)}
        sparse_rank = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(sparse_hits, start=1)}
        dense_score_map = {chunk.chunk_id: score for chunk, score in dense_hits}
        sparse_score_map = {chunk.chunk_id: score for chunk, score in sparse_hits}
        chunk_by_id: dict[str, Chunk] = {}
        for chunk, _ in dense_hits:
            chunk_by_id[chunk.chunk_id] = chunk
        for chunk, _ in sparse_hits:
            chunk_by_id.setdefault(chunk.chunk_id, chunk)

        fused_scores: dict[str, float] = {}
        for chunk_id in chunk_by_id:
            score = 0.0
            if chunk_id in dense_rank:
                score += 1.0 / (self.rrf_k + dense_rank[chunk_id])
            if chunk_id in sparse_rank:
                score += 1.0 / (self.rrf_k + sparse_rank[chunk_id])
            fused_scores[chunk_id] = score

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:fused_top_k]

        return [
            RetrievedChunk(
                chunk=chunk_by_id[cid],
                dense_score=dense_score_map.get(cid),
                sparse_score=sparse_score_map.get(cid),
                fused_score=fused_scores[cid],
            )
            for cid in ranked_ids
        ]
