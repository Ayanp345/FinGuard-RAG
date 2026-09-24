from __future__ import annotations

from pydantic import BaseModel, Field

from src.models import Citation, QueryType


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)


class QueryResponse(BaseModel):
    query: str
    query_type: QueryType
    answer: str
    citations: list[Citation]
    faithfulness_score: float | None
    unsupported_claim_count: int
    retrieved_chunk_ids: list[str]
    latency_ms: float | None
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    llm_backend: str


class ErrorResponse(BaseModel):
    detail: str
