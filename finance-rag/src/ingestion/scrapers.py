from __future__ import annotations

import logging
import time
import urllib.robotparser as robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

logger = logging.getLogger(__name__)

USER_AGENT = "FinanceRAG-Research-Bot/1.0 (+contact: you@example.com)"
REQUEST_DELAY_SECONDS = 2.0


@dataclass
class SourceEntry:
    name: str
    url: str
    doc_type: str


def load_sources(manifest_path: Path) -> list[SourceEntry]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    entries: list[SourceEntry] = []
    for doc_type, items in raw.items():
        for item in items or []:
            entries.append(SourceEntry(name=item["name"], url=item["url"], doc_type=doc_type))
    return entries


def _robots_allow(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        # If robots.txt is unreachable we fail closed and skip the file
        # rather than guessing.
        logger.warning("Could not read robots.txt for %s; skipping to be safe.", url)
        return False


def download_sources(manifest_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for entry in load_sources(manifest_path):
        dest = out_dir / f"{entry.name}.pdf"
        if dest.exists():
            logger.info("Already have %s, skipping", dest.name)
            downloaded.append(dest)
            continue

        if not _robots_allow(entry.url):
            logger.warning("robots.txt disallows fetching %s — skipped.", entry.url)
            continue

        logger.info("Downloading %s", entry.url)
        try:
            resp = session.get(entry.url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            downloaded.append(dest)
        except requests.RequestException as exc:
            logger.error("Failed to download %s: %s", entry.url, exc)

        time.sleep(REQUEST_DELAY_SECONDS)

    return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download curated source PDFs.")
    parser.add_argument("--manifest", default="data/sources.yaml", type=Path)
    parser.add_argument("--out", default="data/raw", type=Path)
    args = parser.parse_args()
    files = download_sources(args.manifest, args.out)
    print(f"Downloaded/verified {len(files)} files into {args.out}")
