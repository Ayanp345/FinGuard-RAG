from src.models import Chunk, DocType
from src.retrieval.hybrid_retriever import HybridRetriever


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        doc_id="doc-1",
        doc_type=DocType.ANNUAL_REPORT,
        source_name="test.pdf",
    )


class _StubRetriever:
    """Mimics DenseRetriever/SparseRetriever's `.search()` interface with a
    fixed, hand-specified ranking so fusion math can be checked exactly."""

    def __init__(self, ranking: list[tuple[str, float]]):
        self._ranking = ranking

    def search(self, query: str, top_k: int = 25):
        return [(_chunk(cid), score) for cid, score in self._ranking[:top_k]]


def test_rrf_favors_chunk_ranked_well_in_both_lists():
    # "b" is #2 in both lists; "a" is #1 dense only; "c" is #1 sparse only.
    dense = _StubRetriever([("a", 0.9), ("b", 0.8), ("d", 0.7)])
    sparse = _StubRetriever([("c", 5.0), ("b", 4.0), ("e", 3.0)])
    hybrid = HybridRetriever(dense, sparse, rrf_k=60)

    results = hybrid.retrieve("query", dense_top_k=10, sparse_top_k=10, fused_top_k=10)
    ranked_ids = [r.chunk.chunk_id for r in results]

    assert ranked_ids[0] == "b"  # consistently ranked #2 in both beats being #1 in only one


def test_rrf_includes_chunks_found_by_only_one_retriever():
    dense = _StubRetriever([("a", 0.9)])
    sparse = _StubRetriever([])
    hybrid = HybridRetriever(dense, sparse, rrf_k=60)

    results = hybrid.retrieve("query", dense_top_k=10, sparse_top_k=10, fused_top_k=10)
    assert [r.chunk.chunk_id for r in results] == ["a"]
    assert results[0].sparse_score is None
    assert results[0].dense_score == 0.9


def test_fused_top_k_caps_result_count():
    dense = _StubRetriever([(f"d{i}", 1.0 - i * 0.01) for i in range(20)])
    sparse = _StubRetriever([(f"s{i}", 20.0 - i) for i in range(20)])
    hybrid = HybridRetriever(dense, sparse, rrf_k=60)

    results = hybrid.retrieve("query", dense_top_k=20, sparse_top_k=20, fused_top_k=5)
    assert len(results) == 5
