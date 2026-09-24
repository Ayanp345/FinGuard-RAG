from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from src.models import ChunkType, DocType

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass
class PageRecord:
    doc_id: str
    source_name: str
    source_url: str | None
    doc_type: DocType
    page_number: int
    text: str
    tables_markdown: list[str] = field(default_factory=list)


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _table_to_markdown(table: list[list[str | None]]) -> str | None:
    """Convert a pdfplumber table (list of rows) into a markdown table string."""
    if not table or len(table) < 2:
        return None
    rows = [[(cell or "").strip() for cell in row] for row in table]
    # Drop fully-empty rows which pdfplumber sometimes emits for merged cells.
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return None
    header, *body = rows
    n_cols = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * n_cols) + " |",
    ]
    for row in body:
        row = (row + [""] * n_cols)[:n_cols]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def infer_doc_type(filename: str) -> DocType:
    name = filename.lower()
    if "rbi" in name or "circular" in name:
        return DocType.RBI_CIRCULAR
    if "sebi" in name or "regulation" in name:
        return DocType.SEBI_REGULATION
    if "budget" in name or "economic_survey" in name:
        return DocType.BUDGET_DOCUMENT
    if "annual" in name or "report" in name:
        return DocType.ANNUAL_REPORT
    return DocType.OTHER


def parse_pdf(
    path: Path,
    doc_type: DocType | None = None,
    source_url: str | None = None,
) -> list[PageRecord]:
    """Parse a single PDF into one PageRecord per page."""
    doc_id = str(uuid.uuid4())
    doc_type = doc_type or infer_doc_type(path.name)
    records: list[PageRecord] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception:  # pdfplumber can choke on malformed pages
                logger.warning("Text extraction failed on %s page %d", path.name, i)
                raw_text = ""

            tables_md: list[str] = []
            try:
                for table in page.extract_tables():
                    md = _table_to_markdown(table)
                    if md:
                        tables_md.append(md)
            except Exception:
                logger.warning("Table extraction failed on %s page %d", path.name, i)

            records.append(
                PageRecord(
                    doc_id=doc_id,
                    source_name=path.name,
                    source_url=source_url,
                    doc_type=doc_type,
                    page_number=i,
                    text=_clean_text(raw_text),
                    tables_markdown=tables_md,
                )
            )
    logger.info("Parsed %s: %d pages", path.name, len(records))
    return records


def parse_directory(directory: Path) -> list[PageRecord]:
    """Parse every PDF in a directory (non-recursive)."""
    all_records: list[PageRecord] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        all_records.extend(parse_pdf(pdf_path))
    return all_records
