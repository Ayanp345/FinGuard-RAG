from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocType(str, Enum):
    ANNUAL_REPORT = "annual_report"
    RBI_CIRCULAR = "rbi_circular"
    SEBI_REGULATION = "sebi_regulation"
    BUDGET_DOCUMENT = "budget_document"
    OTHER = "other"


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"


class Chunk(BaseModel):
    """A single retrievable unit of text (or a serialized table)."""

    chunk_id: str
    text: str
    doc_id: str
    doc_type: DocType = DocType.OTHER
    source_name: str = Field(..., description="e.g. 'TCS_Annual_Report_2024.pdf'")
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_type: ChunkType = ChunkType.TEXT
    token_count: int = 0

    def to_metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type.value,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "page_number": self.page_number,
            "section": self.section,
            "chunk_type": self.chunk_type.value,
        }


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


class QueryType(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    REGULATORY = "regulatory"
    COMPARATIVE = "comparative"


class Citation(BaseModel):
    claim: str
    chunk_id: str
    source_name: str
    page_number: Optional[int] = None
    entailment_score: Optional[float] = Field(
        None, description="NLI entailment probability of `claim` given the cited chunk"
    )
    supported: Optional[bool] = None


class RAGAnswer(BaseModel):
    query: str
    query_type: QueryType
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    faithfulness_score: Optional[float] = None
    unsupported_claim_count: int = 0
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: Optional[float] = None
