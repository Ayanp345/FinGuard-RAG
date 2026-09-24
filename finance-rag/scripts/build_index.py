from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import settings
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_parser import parse_directory
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main(raw_dir: Path, processed_dir: Path, index_dir: Path) -> None:
    logger.info("Parsing PDFs in %s ...", raw_dir)
    pages = parse_directory(raw_dir)
    if not pages:
        logger.error("No PDFs found in %s. Add source PDFs first (see scripts/download_data.py).", raw_dir)
        return
    logger.info("Parsed %d pages total.", len(pages))

    logger.info("Chunking...")
    chunks = chunk_pages(pages, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    logger.info(
        "Produced %d chunks (%d table chunks).",
        len(chunks),
        sum(1 for c in chunks if c.chunk_type.value == "table"),
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    logger.info("Building dense index with %s ...", settings.embedding_model)
    dense = DenseRetriever(settings.embedding_model, device=settings.device)
    dense.build(chunks)
    dense.save(index_dir)

    logger.info("Building sparse (BM25) index ...")
    sparse = SparseRetriever()
    sparse.build(chunks)
    sparse.save(index_dir)

    stats = {
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "n_table_chunks": sum(1 for c in chunks if c.chunk_type.value == "table"),
        "embedding_model": settings.embedding_model,
    }
    with open(index_dir / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Done. Index stats: %s", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(settings.data_raw_dir), type=Path)
    parser.add_argument("--processed-dir", default=str(settings.data_processed_dir), type=Path)
    parser.add_argument("--index-dir", default=str(settings.index_dir), type=Path)
    args = parser.parse_args()
    main(args.raw_dir, args.processed_dir, args.index_dir)
