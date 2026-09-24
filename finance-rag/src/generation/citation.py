"""
Turn the LLM's raw {"answer", "claims": [...]} JSON into validated Citation
objects.

Critically, this step never trusts the model's cited chunk_id at face value:
a model can (and small/quantized models especially will) invent a chunk_id
that was never actually retrieved. Every citation is cross-checked against
the set of chunk_ids that were genuinely handed to the generator; anything
else is dropped and the claim is marked as uncited, which the hallucination
checker downstream will then correctly flag as unsupported rather than
silently trusting a fabricated source.
"""
from __future__ import annotations

import logging

from src.models import Citation, RetrievedChunk

logger = logging.getLogger(__name__)


def extract_citations(
    llm_output: dict, retrieved: list[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    chunk_by_id = {rc.chunk.chunk_id: rc.chunk for rc in retrieved}

    answer = llm_output.get("answer", "")
    raw_claims = llm_output.get("claims", [])
    citations: list[Citation] = []

    for raw_claim in raw_claims:
        text = raw_claim.get("text", "").strip()
        if not text:
            continue
        source_ids = raw_claim.get("source_ids") or []
        valid_ids = [sid for sid in source_ids if sid in chunk_by_id]

        if not valid_ids:
            logger.warning("Claim cited no valid chunk_id (hallucinated citation?): %r", text[:80])
            citations.append(
                Citation(claim=text, chunk_id="", source_name="UNVERIFIED", supported=False)
            )
            continue

        for cid in valid_ids:
            chunk = chunk_by_id[cid]
            citations.append(
                Citation(
                    claim=text,
                    chunk_id=cid,
                    source_name=chunk.source_name,
                    page_number=chunk.page_number,
                )
            )

    return answer, citations
