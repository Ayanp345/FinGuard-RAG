from __future__ import annotations

import logging
import time

from src.config import Settings
from src.generation.citation import extract_citations
from src.generation.llm_client import LLMClient
from src.generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from src.hallucination_checker import HallucinationChecker
from src.models import QueryType, RAGAnswer, RetrievedChunk
from src.query_classifier import QueryClassifier
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        classifier: QueryClassifier,
        llm_client: LLMClient,
        hallucination_checker: HallucinationChecker,
        settings: Settings,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.classifier = classifier
        self.llm_client = llm_client
        self.hallucination_checker = hallucination_checker
        self.settings = settings

    def _retrieve(self, query: str, query_type: QueryType) -> list[RetrievedChunk]:
        s = self.settings

        if query_type == QueryType.COMPARATIVE:
            entities = self.classifier.extract_entities(query)
            if len(entities) >= 2:
                logger.info("Comparative query, multi-hop retrieval for entities: %s", entities)
                seen_ids: set[str] = set()
                merged: list[RetrievedChunk] = []
                # Split the reranker budget across entities so each gets
                # meaningful coverage instead of one dominating.
                per_entity_k = max(3, s.fused_top_k // len(entities))
                for entity in entities:
                    sub_query = f"{entity}: {query}"
                    hits = self.hybrid_retriever.retrieve(
                        sub_query, s.dense_top_k, s.sparse_top_k, per_entity_k
                    )
                    for h in hits:
                        if h.chunk.chunk_id not in seen_ids:
                            merged.append(h)
                            seen_ids.add(h.chunk.chunk_id)
                return merged

        return self.hybrid_retriever.retrieve(query, s.dense_top_k, s.sparse_top_k, s.fused_top_k)

    async def answer(self, query: str) -> RAGAnswer:
        start = time.perf_counter()
        s = self.settings

        query_type = self.classifier.classify(query)
        fused = self._retrieve(query, query_type)
        top_chunks = self.reranker.rerank(query, fused, top_k=s.final_top_k)

        if not top_chunks:
            return RAGAnswer(
                query=query,
                query_type=query_type,
                answer="I couldn't find any relevant information in the indexed documents "
                       "to answer this question.",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        system = SYSTEM_PROMPT
        user = build_generation_prompt(query, top_chunks)

        try:
            llm_output = await self.llm_client.complete_json(
                system, user, max_tokens=s.generation_max_tokens
            )
        except Exception as exc:
            logger.exception("Generation failed: %s", exc)
            return RAGAnswer(
                query=query,
                query_type=query_type,
                answer="The model failed to produce a valid structured answer for this query. "
                       "Please retry.",
                retrieved_chunk_ids=[rc.chunk.chunk_id for rc in top_chunks],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        answer_text, citations = extract_citations(llm_output, top_chunks)
        chunk_by_id = {rc.chunk.chunk_id: rc.chunk for rc in top_chunks}
        citations = self.hallucination_checker.check(
            citations, chunk_by_id, entailment_threshold=s.entailment_threshold
        )
        faithfulness = HallucinationChecker.aggregate_faithfulness(citations)
        unsupported = sum(1 for c in citations if not c.supported)

        return RAGAnswer(
            query=query,
            query_type=query_type,
            answer=answer_text,
            citations=citations,
            faithfulness_score=faithfulness,
            unsupported_claim_count=unsupported,
            retrieved_chunk_ids=list(chunk_by_id.keys()),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
