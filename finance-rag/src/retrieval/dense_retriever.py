"""
Dense retriever: sentence-transformer embeddings + FAISS.

Uses IndexFlatIP over L2-normalized vectors, which is mathematically
equivalent to cosine similarity but lets FAISS use its faster inner-product
path. For corpora beyond a few hundred thousand chunks, swap
`IndexFlatIP` for `IndexHNSWFlat` (approximate, sub-linear) — the public
interface below doesn't change.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import Chunk

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name, device=device)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    # -- building -----------------------------------------------------------
    def build(self, chunks: list[Chunk], batch_size: int = 64, show_progress: bool = True) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        logger.info("Built dense index: %d vectors, dim=%d", index.ntotal, dim)

    # -- persistence ----------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json() + "\n")
        with open(directory / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name}, f)

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> "DenseRetriever":
        with open(directory / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        retriever = cls(model_name=meta["model_name"], device=device)
        retriever.index = faiss.read_index(str(directory / "faiss.index"))
        retriever.chunks = []
        with open(directory / "chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                retriever.chunks.append(Chunk.model_validate_json(line))
        return retriever

    # -- search ---------------------------------------------------------------
    def search(self, query: str, top_k: int = 25) -> list[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Index not built/loaded.")
        q_emb = self.encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def embed_query(self, query: str) -> np.ndarray:
        return self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
