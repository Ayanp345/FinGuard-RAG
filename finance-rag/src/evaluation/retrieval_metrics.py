from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if cid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    results: list[tuple[list[str], set[str]]], k_values: tuple[int, ...] = (5, 10)
) -> dict[str, float]:
    """`results` is a list of (retrieved_chunk_ids_in_rank_order, relevant_chunk_ids)."""
    metrics: dict[str, list[float]] = {}
    for retrieved, relevant in results:
        for k in k_values:
            metrics.setdefault(f"recall@{k}", []).append(recall_at_k(retrieved, relevant, k))
            metrics.setdefault(f"precision@{k}", []).append(precision_at_k(retrieved, relevant, k))
            metrics.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(retrieved, relevant, k))
        metrics.setdefault("mrr", []).append(reciprocal_rank(retrieved, relevant))

    def _mean(values: list[float]) -> float:
        clean = [v for v in values if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else float("nan")

    return {name: round(_mean(values), 4) for name, values in metrics.items()}
