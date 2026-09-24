"""
Evaluation harness.

Usage:
    python -m src.evaluation.run_eval --eval-set data/eval/eval_qa.jsonl \
        --index-dir data/index --out data/eval/report.json

Expects each line of the eval set to be a JSON object:
    {"question": "...", "reference_answer": "...", "relevant_chunk_ids": ["...", ...]}

`relevant_chunk_ids` is used for retrieval metrics (recall/precision/MRR/nDCG)
and must be hand-labeled — see scripts/generate_eval_set.py for a helper that
bootstraps candidates for you to review, not a substitute for review.

Three conditions are compared, matching the original project's ask:
  - no_rag:   the LLM answers from parametric knowledge alone (no context)
  - naive_rag: single dense-retrieval hit, no reranking, no citation checking
  - full_pipeline: hybrid retrieval + RRF + rerank + citations + NLI check
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path

from src.config import settings
from src.evaluation.correctness import judge_correctness
from src.evaluation.relevance import compute_answer_relevance
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.generation.citation import extract_citations
from src.generation.llm_client import build_llm_client
from src.generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from src.hallucination_checker import HallucinationChecker
from src.models import RetrievedChunk
from src.pipeline import RAGPipeline
from src.query_classifier import QueryClassifier
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse_retriever import SparseRetriever

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


def load_eval_set(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def run_no_rag(llm_client, question: str) -> tuple[str, float]:
    start = time.perf_counter()
    try:
        result = await llm_client.complete_json(
            "Answer the financial question as best you can from your own knowledge. "
            'Respond with ONLY {"answer": "...", "claims": []}.',
            question,
            max_tokens=400,
        )
        answer = result.get("answer", "")
    except Exception:
        answer = ""
    return answer, (time.perf_counter() - start) * 1000


async def run_naive_rag(dense: DenseRetriever, llm_client, question: str) -> tuple[str, list[str], float]:
    start = time.perf_counter()
    hits = dense.search(question, top_k=1)
    if not hits:
        return "No relevant context found.", [], (time.perf_counter() - start) * 1000
    chunk, _ = hits[0]
    rc = [RetrievedChunk(chunk=chunk)]
    try:
        result = await llm_client.complete_json(SYSTEM_PROMPT, build_generation_prompt(question, rc), 400)
        answer, _ = extract_citations(result, rc)
    except Exception:
        answer = ""
    return answer, [chunk.chunk_id], (time.perf_counter() - start) * 1000


async def main(args: argparse.Namespace) -> None:
    eval_set = load_eval_set(Path(args.eval_set))
    index_dir = Path(args.index_dir)

    dense = DenseRetriever.load(index_dir, device=settings.device)
    sparse = SparseRetriever.load(index_dir)
    hybrid = HybridRetriever(dense, sparse, rrf_k=settings.rrf_k)
    reranker = Reranker(settings.reranker_model, settings.device)
    classifier = QueryClassifier(dense.encoder)
    hallucination_checker = HallucinationChecker(settings.nli_model, settings.device)
    llm_client = build_llm_client(settings)
    pipeline = RAGPipeline(hybrid, reranker, classifier, llm_client, hallucination_checker, settings)

    retrieval_pairs: list[tuple[list[str], set[str]]] = []
    per_condition_latency: dict[str, list[float]] = {"no_rag": [], "naive_rag": [], "full_pipeline": []}
    correctness_scores: dict[str, list[float]] = {"no_rag": [], "naive_rag": [], "full_pipeline": []}
    relevance_scores: dict[str, list[float]] = {"no_rag": [], "naive_rag": [], "full_pipeline": []}
    faithfulness_scores: list[float] = []

    judge_client = llm_client  # reuse; swap for a stronger model if desired

    for item in eval_set:
        question = item["question"]
        reference = item["reference_answer"]
        relevant_ids = set(item.get("relevant_chunk_ids", []))
        logger.info("Evaluating: %s", question)

        no_rag_answer, no_rag_latency = await run_no_rag(llm_client, question)
        per_condition_latency["no_rag"].append(no_rag_latency)

        naive_answer, naive_ids, naive_latency = await run_naive_rag(dense, llm_client, question)
        per_condition_latency["naive_rag"].append(naive_latency)
        if relevant_ids:
            retrieval_pairs.append((naive_ids, relevant_ids))

        full_result = await pipeline.answer(question)
        per_condition_latency["full_pipeline"].append(full_result.latency_ms or 0.0)
        if relevant_ids:
            retrieval_pairs.append((full_result.retrieved_chunk_ids, relevant_ids))
        if full_result.faithfulness_score is not None:
            faithfulness_scores.append(full_result.faithfulness_score)

        for condition, answer in [
            ("no_rag", no_rag_answer),
            ("naive_rag", naive_answer),
            ("full_pipeline", full_result.answer),
        ]:
            score, _ = await judge_correctness(question, reference, answer, judge_client)
            if score == score:  # filter NaN
                correctness_scores[condition].append(score)
            rel = await compute_answer_relevance(question, answer, llm_client, dense.encoder)
            if rel == rel:
                relevance_scores[condition].append(rel)

    def _avg(values: list[float]) -> float | None:
        return round(statistics.mean(values), 4) if values else None

    report = {
        "n_questions": len(eval_set),
        "retrieval_metrics_naive_vs_full": evaluate_retrieval(retrieval_pairs) if retrieval_pairs else {},
        "faithfulness_full_pipeline": _avg(faithfulness_scores),
        "correctness_by_condition": {k: _avg(v) for k, v in correctness_scores.items()},
        "relevance_by_condition": {k: _avg(v) for k, v in relevance_scores.items()},
        "mean_latency_ms_by_condition": {k: _avg(v) for k, v in per_condition_latency.items()},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the no-RAG vs naive-RAG vs full-pipeline evaluation.")
    parser.add_argument("--eval-set", default="data/eval/eval_qa.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--out", default="data/eval/report.json")
    asyncio.run(main(parser.parse_args()))
