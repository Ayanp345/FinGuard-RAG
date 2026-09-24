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
