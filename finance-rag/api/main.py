from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.deps import AppState, get_pipeline, verify_api_key
from api.schemas import ErrorResponse, HealthResponse, QueryRequest, QueryResponse
from src.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading indices and models (this can take a while on first boot)...")
    state = AppState()
    state.load(settings)
    app.state.app_state = state
    logger.info("Startup complete. LLM backend: %s", settings.llm_backend)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Finance RAG API",
    description="RAG over Indian financial documents with hybrid retrieval and citation verification.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state: AppState = request.app.state.app_state
    return HealthResponse(
        status="ok",
        index_loaded=state.pipeline is not None,
        llm_backend=settings.llm_backend,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    state: AppState = request.app.state.app_state
    pipeline = get_pipeline(request)

    cached = await state.get_cached(body.query)
    if cached:
        return QueryResponse(**cached, cached=True)

    result = await pipeline.answer(body.query)
    response = QueryResponse(
        query=result.query,
        query_type=result.query_type,
        answer=result.answer,
        citations=result.citations,
        faithfulness_score=result.faithfulness_score,
        unsupported_claim_count=result.unsupported_claim_count,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        latency_ms=result.latency_ms,
    )
    await state.set_cached(body.query, response.model_dump(exclude={"cached"}))
    return response


@app.post("/query/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit(settings.rate_limit)
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    pipeline = get_pipeline(request)

    async def event_generator():
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("stage", {"stage": "classifying"})
        query_type = pipeline.classifier.classify(body.query)
        yield sse("stage", {"stage": "retrieving", "query_type": query_type.value})

        fused = pipeline._retrieve(body.query, query_type)
        yield sse("stage", {"stage": "reranking", "n_candidates": len(fused)})

        top_chunks = pipeline.reranker.rerank(body.query, fused, top_k=pipeline.settings.final_top_k)
        yield sse("stage", {"stage": "generating"})

        result = await pipeline.answer(body.query)  # re-runs classify+retrieve; simplicity over
        # micro-optimizing a duplicate embedding call, which is cheap relative to generation.
        yield sse("stage", {"stage": "verifying_citations"})
        yield sse("final", result.model_dump())

    return StreamingResponse(event_generator(), media_type="text/event-stream")
