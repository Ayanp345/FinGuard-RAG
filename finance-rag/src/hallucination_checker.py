"""
NLI-based hallucination detection.

String-matching ("does the number in the claim appear in the chunk") is
what the original naive plan used, and it's easy to fool: a claim can be
lexically grounded but semantically wrong ("revenue grew" vs "revenue fell"
both contain the word "revenue" and a percentage). Instead, each (chunk,
claim) pair is scored by a cross-encoder NLI model treating the chunk as
the premise and the claim as the hypothesis. The label of interest is
`entailment`: does the source text actually entail the claim? Contradiction
or neutral both mean the claim isn't supported, even if it shares
vocabulary with the source.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from src.models import Chunk, Citation

# cross-encoder/nli-deberta-v3-base label order.
_LABELS = ["contradiction", "entailment", "neutral"]


class HallucinationChecker:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def check(
        self,
        citations: list[Citation],
        chunk_by_id: dict[str, Chunk],
        entailment_threshold: float = 0.5,
    ) -> list[Citation]:
        pairs, indices = [], []
        for i, citation in enumerate(citations):
            chunk = chunk_by_id.get(citation.chunk_id)
            if chunk is None:
                citation.supported = False
                citation.entailment_score = 0.0
                continue
            pairs.append((chunk.text, citation.claim))
            indices.append(i)

        if pairs:
            logits = self.model.predict(pairs)
            probs = _softmax(logits)
            entail_idx = _LABELS.index("entailment")
            for i, prob_row in zip(indices, probs):
                entail_prob = float(prob_row[entail_idx])
                citations[i].entailment_score = entail_prob
                citations[i].supported = entail_prob >= entailment_threshold

        return citations

    @staticmethod
    def aggregate_faithfulness(citations: list[Citation]) -> float:
        """Fraction of claims that are NLI-supported by their cited source."""
        if not citations:
            return 1.0  # no factual claims made => vacuously faithful
        supported = sum(1 for c in citations if c.supported)
        return supported / len(citations)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)
