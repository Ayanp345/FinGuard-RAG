from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from src.config import settings
from src.generation.llm_client import build_llm_client
from src.retrieval.dense_retriever import DenseRetriever

QA_GEN_PROMPT = """Given the following excerpt from an Indian financial document, write ONE \
specific factual question that can be answered using ONLY this excerpt, plus the correct answer.

Excerpt:
{chunk_text}

Respond with ONLY a JSON object: {{"question": "...", "answer": "..."}}"""


async def main(n: int, out_path: Path, seed: int) -> None:
    random.seed(seed)
    dense = DenseRetriever.load(Path(settings.index_dir), device=settings.device)
    llm_client = build_llm_client(settings)

    # Prefer table chunks and longer text chunks: they tend to contain the
    # concrete, checkable figures that make good eval questions.
    candidates = [c for c in dense.chunks if c.token_count and c.token_count > 40]
    if len(candidates) < n:
        raise SystemExit(f"Only {len(candidates)} eligible chunks indexed; requested {n}.")
    sampled = random.sample(candidates, n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in sampled:
            try:
                result = await llm_client.complete_json(
                    "You write factual questions for evaluating a financial document QA system.",
                    QA_GEN_PROMPT.format(chunk_text=chunk.text),
                    max_tokens=250,
                )
                record = {
                    "question": result["question"],
                    "reference_answer": result["answer"],
                    "relevant_chunk_ids": [chunk.chunk_id],
                    "source_name": chunk.source_name,
                    "page_number": chunk.page_number,
                    "REVIEWED": False,
                }
                f.write(json.dumps(record) + "\n")
                written += 1
            except Exception as exc:
                print(f"Skipped one chunk due to generation error: {exc}")

    print(f"Wrote {written} CANDIDATE pairs to {out_path}.")
    print("These are unreviewed — verify each one against the source PDF before using them "
          "as ground truth in run_eval.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data/eval/eval_qa_candidates.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args.n, args.out, args.seed))
