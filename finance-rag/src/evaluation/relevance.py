"""
Answer relevance: does the answer actually address the question asked?

Faithfulness (hallucination_checker.py) tells you whether claims are
grounded in the retrieved context; it says nothing about whether the model
answered a *different, easier* question than the one asked (a common
failure mode when retrieved context is thin). This metric follows the
RAGAS approach: ask the generator to produce N plausible questions that
the answer *does* address, then score relevance as the mean cosine
similarity between the original question and those reverse-engineered
questions. A model that answered off-topic will generate reverse questions
that drift from the original.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.generation.llm_client import LLMClient

REVERSE_QUESTION_PROMPT = """Given the ANSWER below, generate {n} questions that this answer \
would be a good response to. Respond with ONLY a JSON object: {{"questions": ["...", ...]}}

ANSWER:
{answer}"""


async def compute_answer_relevance(
    question: str,
    answer: str,
    llm_client: LLMClient,
    encoder: SentenceTransformer,
    n_reverse_questions: int = 3,
) -> float:
    if not answer.strip():
        return 0.0

    prompt = REVERSE_QUESTION_PROMPT.format(n=n_reverse_questions, answer=answer)
    try:
        result = await llm_client.complete_json(
            "You generate reverse-engineered questions for RAG evaluation.", prompt, max_tokens=300
        )
        reverse_questions = result.get("questions", [])
    except Exception:
        reverse_questions = []

    if not reverse_questions:
        return float("nan")

    q_emb = encoder.encode([question], normalize_embeddings=True)[0]
    rq_embs = encoder.encode(reverse_questions, normalize_embeddings=True)
    similarities = rq_embs @ q_emb
    return float(np.mean(similarities))
