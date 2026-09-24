from __future__ import annotations

import re
import uuid

import tiktoken

from src.ingestion.pdf_parser import PageRecord
from src.models import Chunk, ChunkType

_ENCODING = tiktoken.get_encoding("cl100k_base")

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\b[A-Z])(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bRs)(?<=[.?!])\s+(?=[A-Z(\u20b9])"
)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_sentences(paragraph: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]


def _pack_sentences(
    sentences: list[str], max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Greedily pack sentences into token-bounded windows with overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        # A single sentence longer than the whole budget: hard-split on words.
        if sent_tokens > max_tokens:
            words = sentence.split()
            buf: list[str] = []
            buf_tokens = 0
            for word in words:
                wt = count_tokens(word + " ")
                if buf_tokens + wt > max_tokens and buf:
                    chunks.append(" ".join(buf))
                    buf, buf_tokens = [], 0
                buf.append(word)
                buf_tokens += wt
            if buf:
                sentence_pieces = [" ".join(buf)]
            else:
                sentence_pieces = []
            for piece in sentence_pieces:
                if current_tokens + count_tokens(piece) > max_tokens and current:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                current.append(piece)
                current_tokens += count_tokens(piece)
            continue

        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            # Build overlap: keep trailing sentences up to overlap budget.
            overlap: list[str] = []
            overlap_tok = 0
            for s in reversed(current):
                t = count_tokens(s)
                if overlap_tok + t > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_tok += t
            current, current_tokens = list(overlap), overlap_tok

        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_page(
    page: PageRecord, max_tokens: int = 350, overlap_tokens: int = 60
) -> list[Chunk]:
    chunks: list[Chunk] = []

    # 1. Tables: one chunk each, never split.
    for table_md in page.tables_markdown:
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=table_md,
                doc_id=page.doc_id,
                doc_type=page.doc_type,
                source_name=page.source_name,
                source_url=page.source_url,
                page_number=page.page_number,
                chunk_type=ChunkType.TABLE,
                token_count=count_tokens(table_md),
            )
        )

    # 2. Prose: paragraph -> sentence -> token-bounded packing.
    paragraphs = [p for p in _PARAGRAPH_SPLIT_RE.split(page.text) if p.strip()]
    all_sentences: list[str] = []
    for para in paragraphs:
        all_sentences.extend(_split_sentences(para))

    if all_sentences:
        for text in _pack_sentences(all_sentences, max_tokens, overlap_tokens):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=text,
                    doc_id=page.doc_id,
                    doc_type=page.doc_type,
                    source_name=page.source_name,
                    source_url=page.source_url,
                    page_number=page.page_number,
                    chunk_type=ChunkType.TEXT,
                    token_count=count_tokens(text),
                )
            )

    return chunks


def chunk_pages(
    pages: list[PageRecord], max_tokens: int = 350, overlap_tokens: int = 60
) -> list[Chunk]:
    out: list[Chunk] = []
    for page in pages:
        out.extend(chunk_page(page, max_tokens, overlap_tokens))
    return out
