from __future__ import annotations

import re

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import QueryType

EXEMPLARS: dict[QueryType, list[str]] = {
    QueryType.FACTUAL: [
        "What was TCS revenue in FY2024?",
        "What is Infosys's net profit for the last quarter?",
        "How many employees does Reliance have?",
        "What was HDFC Bank's total assets last year?",
    ],
    QueryType.ANALYTICAL: [
        "Why did Infosys margins decline this year?",
        "What are the key risks mentioned in the annual report?",
        "Summarize the management discussion and analysis section.",
        "What factors drove revenue growth for TCS?",
    ],
    QueryType.REGULATORY: [
        "What does the RBI circular say about NPA classification?",
        "What are SEBI's disclosure requirements for listed companies?",
        "What is the minimum capital adequacy ratio required by RBI?",
        "Summarize the SEBI insider trading regulations.",
    ],
    QueryType.COMPARATIVE: [
        "Compare HDFC Bank and ICICI Bank's net interest margins.",
        "How does TCS's revenue growth compare to Infosys?",
        "Reliance vs Adani: which had higher revenue growth?",
        "Compare the capital adequacy ratios of the top three private banks.",
    ],
}

_COMPARE_TRIGGER_RE = re.compile(r"\b(vs\.?|versus|compare[d]?|compared to)\b", re.IGNORECASE)
# Very rough entity heuristic: capitalized multi-word runs, tuned for the
# domain (Indian company / regulator names). Replace with a proper NER
# model (e.g. spaCy + a finance gazetteer) for production use.
_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z&]*(?:\s+[A-Z][a-zA-Z&]*){0,2})\b")
_STOPWORD_ENTITIES = {"What", "How", "Why", "Compare", "Summarize", "The"}


class QueryClassifier:
    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        self._centroids: dict[QueryType, np.ndarray] = {}
        self._fit_centroids()

    def _fit_centroids(self) -> None:
        for qtype, examples in EXEMPLARS.items():
            embeddings = self.encoder.encode(examples, normalize_embeddings=True)
            self._centroids[qtype] = np.mean(embeddings, axis=0)

    def classify(self, query: str) -> QueryType:
        # A literal "vs"/"compare" is a strong enough signal to short-circuit
        # the (noisier) embedding vote.
        if _COMPARE_TRIGGER_RE.search(query):
            return QueryType.COMPARATIVE

        q_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
        best_type, best_score = None, -1.0
        for qtype, centroid in self._centroids.items():
            score = float(np.dot(q_emb, centroid))
            if score > best_score:
                best_type, best_score = qtype, score
        return best_type or QueryType.FACTUAL

    @staticmethod
    def extract_entities(query: str) -> list[str]:
        """Pull out likely named entities for multi-hop comparative retrieval."""
        candidates = _ENTITY_RE.findall(query)
        entities = [c for c in candidates if c not in _STOPWORD_ENTITIES and len(c) > 2]
        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique = []
        for e in entities:
            if e not in seen:
                unique.append(e)
                seen.add(e)
        return unique
