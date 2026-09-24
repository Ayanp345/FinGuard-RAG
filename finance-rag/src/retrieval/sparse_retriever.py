"""
BM25 sparse retriever.

Standard whitespace/lowercase tokenizers destroy financial notation:
"Rs. 1,23,456 crore" or "12.5%" lose their meaning once punctuation is
stripped and numbers get shattered. The tokenizer here keeps numbers
(including Indian lakh/crore-style comma grouping), currency symbols and
percent signs attached to their digits, so a query like "revenue grew 12.5%"
can still match lexically against the source text — which dense embeddings
alone are often surprisingly bad at for exact figures.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.models import Chunk

_TOKEN_RE = re.compile(
    r"""
    [₹$€%]?\d[\d,]*\.?\d*%?   # numbers, optionally prefixed by a currency
                              # symbol and/or suffixed by a percent sign,
                              # keeping comma grouping (1,23,456) intact
    | [A-Za-z]+(?:[-'][A-Za-z]+)*   # words, incl. hyphenated / possessive
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class SparseRetriever:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(directory / "bm25_chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json() + "\n")

    @classmethod
    def load(cls, directory: Path) -> "SparseRetriever":
        retriever = cls()
        with open(directory / "bm25.pkl", "rb") as f:
            retriever.bm25 = pickle.load(f)
        retriever.chunks = []
        with open(directory / "bm25_chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                retriever.chunks.append(Chunk.model_validate_json(line))
        return retriever

    def search(self, query: str, top_k: int = 25) -> list[tuple[Chunk, float]]:
        if self.bm25 is None:
            raise RuntimeError("Index not built/loaded.")
        scores = self.bm25.get_scores(tokenize(query))
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx if scores[i] > 0]
