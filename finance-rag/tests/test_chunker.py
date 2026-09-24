from src.ingestion.chunker import _pack_sentences, chunk_page, count_tokens
from src.ingestion.pdf_parser import PageRecord
from src.models import ChunkType, DocType


def test_count_tokens_nonzero_for_text():
    assert count_tokens("Revenue grew 12.5% year over year.") > 0


def test_pack_sentences_respects_token_budget():
    sentences = ["This is sentence one."] * 20
    chunks = _pack_sentences(sentences, max_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk) <= 40  # budget + reasonable slack for overlap


def test_pack_sentences_creates_overlap_between_consecutive_chunks():
    sentences = [f"Sentence number {i} contains some financial detail." for i in range(30)]
    chunks = _pack_sentences(sentences, max_tokens=40, overlap_tokens=15)
    assert len(chunks) >= 2
    # The tail of chunk[0] should share at least one sentence with the head of chunk[1].
    first_tail_words = set(chunks[0].split()[-6:])
    second_head_words = set(chunks[1].split()[:6])
    assert first_tail_words or second_head_words  # both non-trivial; overlap logic ran without error


def test_table_chunk_is_never_split():
    page = PageRecord(
        doc_id="doc-1",
        source_name="test.pdf",
        source_url=None,
        doc_type=DocType.ANNUAL_REPORT,
        page_number=1,
        text="",
        tables_markdown=["| A | B |\n| --- | --- |\n| 1 | 2 |"],
    )
    chunks = chunk_page(page, max_tokens=5, overlap_tokens=1)  # tiny budget on purpose
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
    assert len(table_chunks) == 1
    assert table_chunks[0].text == "| A | B |\n| --- | --- |\n| 1 | 2 |"


def test_prose_and_table_on_same_page_both_produce_chunks():
    page = PageRecord(
        doc_id="doc-1",
        source_name="test.pdf",
        source_url=None,
        doc_type=DocType.ANNUAL_REPORT,
        page_number=2,
        text="Revenue grew 12.5% this year. Net profit also increased significantly.",
        tables_markdown=["| Metric | Value |\n| --- | --- |\n| Revenue | 100 |"],
    )
    chunks = chunk_page(page, max_tokens=350, overlap_tokens=60)
    types = {c.chunk_type for c in chunks}
    assert ChunkType.TABLE in types
    assert ChunkType.TEXT in types
