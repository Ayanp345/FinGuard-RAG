"""Answer correctness via LLM-as-judge against a human-verified reference answer.

Deliberately kept separate from `relevance.py` (on-topic-ness) and the NLI
faithfulness score (groundedness) — an answer can be on-topic and fully
grounded in retrieved text while still being *wrong* if the retrieved chunk
itself doesn't contain the figure the question asked for and the model
plausibly filled the gap. Correctness needs a ground-truth reference and is
therefore the most expensive metric to collect (requires the manually
curated Q&A set from scripts/generate_eval_set.py).
"""
from __future__ import annotations

import logging

from src.generation.llm_client import LLMClient
from src.generation.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt

logger = logging.getLogger(__name__)


async def judge_correctness(
    question: str, reference_answer: str, model_answer: str, judge_client: LLMClient
) -> tuple[float, str]:
    prompt = build_judge_prompt(question, reference_answer, model_answer)
    try:
        result = await judge_client.complete_json(JUDGE_SYSTEM_PROMPT, prompt, max_tokens=200)
        return float(result["score"]), str(result.get("reasoning", ""))
    except Exception as exc:
        logger.warning("Judge call failed for question=%r: %s", question[:60], exc)
        return float("nan"), "judge_error"
