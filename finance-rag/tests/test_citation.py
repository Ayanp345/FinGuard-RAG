from src.generation.citation import extract_citations
from src.models import Chunk, DocType, RetrievedChunk


def _retrieved(chunk_id: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        text=f"source text for {chunk_id}",
        doc_id="doc-1",
        doc_type=DocType.ANNUAL_REPORT,
        source_name="report.pdf",
        page_number=3,
    )
    return RetrievedChunk(chunk=chunk)


def test_valid_citation_is_kept():
    retrieved = [_retrieved("c1")]
    llm_output = {
        "answer": "Revenue grew 10%.",
        "claims": [{"text": "Revenue grew 10%.", "source_ids": ["c1"]}],
    }
    answer, citations = extract_citations(llm_output, retrieved)
    assert answer == "Revenue grew 10%."
    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"
    assert citations[0].source_name == "report.pdf"
    assert citations[0].page_number == 3


def test_hallucinated_chunk_id_is_dropped_and_marked_unverified():
    retrieved = [_retrieved("c1")]
    llm_output = {
        "answer": "Revenue grew 50%.",
        "claims": [{"text": "Revenue grew 50%.", "source_ids": ["c_does_not_exist"]}],
    }
    _, citations = extract_citations(llm_output, retrieved)
    assert len(citations) == 1
    assert citations[0].chunk_id == ""
    assert citations[0].source_name == "UNVERIFIED"
    assert citations[0].supported is False


def test_claim_with_multiple_valid_sources_produces_multiple_citations():
    retrieved = [_retrieved("c1"), _retrieved("c2")]
    llm_output = {
        "answer": "Both banks reported growth.",
        "claims": [{"text": "Both banks reported growth.", "source_ids": ["c1", "c2"]}],
    }
    _, citations = extract_citations(llm_output, retrieved)
    assert {c.chunk_id for c in citations} == {"c1", "c2"}


def test_empty_claims_list_yields_no_citations():
    retrieved = [_retrieved("c1")]
    llm_output = {"answer": "No specific claims here.", "claims": []}
    answer, citations = extract_citations(llm_output, retrieved)
    assert answer == "No specific claims here."
    assert citations == []
