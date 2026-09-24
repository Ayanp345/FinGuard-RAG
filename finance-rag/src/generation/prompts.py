from __future__ import annotations

from src.models import RetrievedChunk

SYSTEM_PROMPT = """You are a financial research assistant answering questions about \
Indian financial documents (company annual reports, RBI circulars, SEBI regulations, \
Union Budget documents).

Rules:
1. Answer ONLY using the numbered source excerpts provided. Do not use outside knowledge.
2. If the excerpts do not contain enough information to answer, say so explicitly instead \
of guessing.
3. Every factual claim in your answer must be traceable to one specific source excerpt.
4. Respond with ONLY a JSON object (no markdown fences, no commentary) matching this schema:
{
  "answer": "<the full prose answer>",
  "claims": [
    {"text": "<a single factual sentence/claim from the answer>", "source_ids": ["<chunk_id>", ...]}
  ]
}
Each entry in "claims" should be one atomic claim from "answer" and the chunk_id(s) of the \
excerpt(s) that support it. If a sentence is connective/non-factual (e.g. "In summary,"), omit it \
from "claims"."""


def build_generation_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    excerpt_blocks = []
    for rc in chunks:
        c = rc.chunk
        loc = f"{c.source_name}, page {c.page_number}" if c.page_number else c.source_name
        excerpt_blocks.append(f"[chunk_id={c.chunk_id}] ({loc})\n{c.text}")
    excerpts = "\n\n".join(excerpt_blocks)
    return (
        f"Question: {query}\n\n"
        f"Source excerpts:\n{excerpts}\n\n"
        "Respond with the JSON object described in the system prompt, and nothing else."
    )


JUDGE_SYSTEM_PROMPT = """You are grading the correctness of an AI-generated answer against a \
human-verified reference answer for a question about Indian financial documents.

Score correctness from 0.0 to 1.0:
- 1.0: fully correct and complete relative to the reference
- 0.5: partially correct, missing key details, or partially inaccurate
- 0.0: incorrect or contradicts the reference

Respond with ONLY a JSON object: {"score": <float 0-1>, "reasoning": "<one sentence>"}"""


def build_judge_prompt(question: str, reference_answer: str, model_answer: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"Model answer: {model_answer}\n\n"
        "Respond with the JSON object described in the system prompt, and nothing else."
    )
