# Finance RAG — Advanced RAG System for Indian Financial Documents

Production-grade Retrieval-Augmented Generation over Indian financial disclosures
(company annual reports, RBI circulars, SEBI regulations, Union Budget documents),
with hybrid retrieval, cross-encoder reranking, NLI-based hallucination detection,
and a real evaluation harness — not just a "ChatPDF" wrapper.

---

## 1. What makes this "advanced" (vs. a naive RAG demo)

| Naive RAG | This system |
|---|---|
| Dense retrieval only | **Hybrid**: dense (FAISS) + sparse (BM25) fused with **Reciprocal Rank Fusion**, not a hand-tuned score blend |
| Top-k dense hits go straight to the LLM | A **cross-encoder reranker** re-scores the fused candidates for query-chunk relevance before generation |
| One query embedding for everything | **Multi-hop retrieval** for comparative queries ("HDFC vs ICICI") — entities are extracted and retrieved separately, then merged, so neither side starves the other |
| Tables flattened into unreadable text | PDF tables are extracted and kept as **atomic markdown chunks**, never split, never mixed with prose |
| Numbers get mangled by tokenizers | Custom BM25 tokenizer preserves `₹1,23,456`-style figures and `12.5%` as intact tokens |
| Hallucination check = "does the number appear somewhere" | **NLI entailment** scoring: does the source chunk actually *entail* the claim, not just share vocabulary with it |
| Citations trusted at face value | Every cited `chunk_id` is **cross-checked against what was actually retrieved** — a model inventing a source is caught, not trusted |
| One correctness metric | Three independent axes: **faithfulness** (NLI groundedness), **relevance** (on-topic-ness via reverse-question embedding similarity), **correctness** (LLM-as-judge vs. human-reviewed reference) |
| Fixed to one open-source model | LLM generation is behind one interface with **three swappable backends** (local HF, Anthropic, OpenAI) — change one env var |
| No caching, no auth, no rate limits | FastAPI backend with **API-key auth, rate limiting, Redis response caching**, and a progress-streaming endpoint |

---

## 2. Architecture

```
                         ┌─────────────────────────┐
   User query  ────────► │   Query Classifier       │  embedding nearest-centroid
                         │  factual/analytical/     │  (+ regex fast-path for
                         │  regulatory/comparative  │   "X vs Y" comparatives)
                         └────────────┬─────────────┘
                                      │
                     comparative? ────┴──── extract entities, retrieve per-entity
                                      │
                         ┌────────────▼─────────────┐
                         │   Hybrid Retriever        │
                         │  FAISS (dense) + BM25     │
                         │  fused via Reciprocal     │
                         │  Rank Fusion              │
                         └────────────┬─────────────┘
                                      │ ~15 candidates
                         ┌────────────▼─────────────┐
                         │   Cross-Encoder Reranker  │  BAAI/bge-reranker-base
                         └────────────┬─────────────┘
                                      │ top 5
                         ┌────────────▼─────────────┐
                         │   Generator (LLM)         │  hf_local | anthropic | openai
                         │  forced structured JSON:  │
                         │  {answer, claims[]}       │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │  Citation Validator       │  drops any cited chunk_id
                         │                           │  that wasn't actually retrieved
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │  NLI Hallucination Check  │  cross-encoder/nli-deberta-v3
                         │  entailment(chunk, claim) │  -> faithfulness score
                         └────────────┬─────────────┘
                                      │
                              RAGAnswer (JSON) ──► FastAPI ──► Streamlit UI
```

### Project layout

```
finance-rag/
├── src/
│   ├── config.py              # env-driven settings (pydantic-settings)
│   ├── models.py              # shared Pydantic schemas (Chunk, Citation, RAGAnswer, ...)
│   ├── pipeline.py            # end-to-end orchestration + multi-hop comparative logic
│   ├── query_classifier.py    # embedding nearest-centroid classifier + entity extraction
│   ├── hallucination_checker.py
│   ├── ingestion/
│   │   ├── pdf_parser.py      # pdfplumber: text + tables, per-page metadata
│   │   ├── chunker.py         # token-aware, table-preserving, sentence-safe chunking
│   │   └── scrapers.py        # polite, robots.txt-respecting downloader for curated URLs
│   ├── retrieval/
│   │   ├── dense_retriever.py # sentence-transformers + FAISS
│   │   ├── sparse_retriever.py# BM25 with a finance-aware tokenizer
│   │   ├── hybrid_retriever.py# Reciprocal Rank Fusion
│   │   └── reranker.py        # cross-encoder reranking
│   ├── generation/
│   │   ├── llm_client.py      # pluggable hf_local / anthropic / openai backends
│   │   ├── prompts.py         # system/user prompt templates
│   │   └── citation.py        # parses + validates the model's cited sources
│   └── evaluation/
│       ├── retrieval_metrics.py  # Recall@k, Precision@k, MRR, nDCG
│       ├── relevance.py          # reverse-question embedding similarity
│       ├── correctness.py        # LLM-as-judge vs. reference answer
│       └── run_eval.py           # CLI: no-RAG vs naive-RAG vs full pipeline
├── api/
│   ├── main.py                 # FastAPI app: /query, /query/stream (SSE), /health
│   ├── deps.py                 # startup model loading, auth, Redis caching
│   └── schemas.py              # request/response models
├── frontend/
│   └── app.py                  # Streamlit chat UI
├── scripts/
│   ├── build_index.py          # PDFs -> chunks -> FAISS + BM25 indices
│   └── generate_eval_set.py    # bootstraps CANDIDATE eval Q&A pairs for human review
├── tests/                      # pytest: chunking, RRF fusion, citation validation, API
├── data/
│   ├── sources.yaml            # curated manifest of source PDF URLs (you fill this in)
│   ├── raw/                    # downloaded PDFs land here
│   ├── processed/              # chunked JSONL
│   ├── index/                  # FAISS + BM25 index files
│   └── eval/                   # eval Q&A sets and reports
├── Dockerfile.backend / Dockerfile.frontend / docker-compose.yml
├── .github/workflows/ci.yml    # lint (ruff) + pytest on every push
├── requirements.txt / .env.example / pytest.ini
```

---

## 3. Setup

### 3.1 Prerequisites

- Python 3.11+
- ~4 GB free disk for model weights (embedding + reranker + NLI models; more if
  you use the local Llama backend)
- Optional: a GPU (set `DEVICE=cuda` in `.env`) — everything runs on CPU by default,
  just slower
- Docker + Docker Compose, if you want the containerized deployment

### 3.2 Install

```bash
git clone <this-repo>
cd finance-rag
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cp .env.example .env
```

Open `.env` and pick a generation backend:

- **`LLM_BACKEND=hf_local`** (default, fully self-hosted, matches the original
  "no external API dependency" resume story): request access to
  [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  on HuggingFace, create a token, and set `HF_TOKEN`. Needs a GPU to be fast;
  runs on CPU but slowly.
- **`LLM_BACKEND=anthropic`** or **`LLM_BACKEND=openai`**: set the matching API
  key. Small open models are noticeably less reliable at emitting the strict
  `{answer, claims[]}` JSON schema the hallucination checker depends on, so a
  hosted model is worth considering even though it's not the "fully
  self-hosted" story — you can mention both in the resume line ("supports
  both self-hosted and hosted generation backends").

### 3.3 Get source documents

This project deliberately does **not** auto-crawl NSE/BSE/RBI/SEBI — see the
comment at the top of `src/ingestion/scrapers.py` for why. Instead:

1. Browse the sites yourself and collect direct PDF links:
   - Annual reports: nseindia.com / bseindia.com / a company's own investor-relations page
   - RBI circulars: rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx
   - SEBI regulations: sebi.gov.in/legal/regulations.html
   - Union Budget: indiabudget.gov.in
2. Fill them into `data/sources.yaml` (replace the placeholder URLs).
3. Download them:
   ```bash
   python -m src.ingestion.scrapers --manifest data/sources.yaml --out data/raw
   ```
   This respects each site's `robots.txt` and rate-limits itself — it will
   politely skip anything disallowed rather than guess.

   Alternatively, just drop PDFs you already have straight into `data/raw/`.

### 3.4 Build the indices

```bash
python -m scripts.build_index --raw-dir data/raw --index-dir data/index
```

This parses every PDF (text + tables separately), chunks it, embeds it with
`BAAI/bge-small-en-v1.5`, and writes both a FAISS index and a BM25 index to
`data/index/`. Check `data/index/build_stats.json` afterwards — if
`n_table_chunks` is 0 for documents you know contain tables, pdfplumber likely
needs a different table-extraction strategy for that PDF's layout (scanned
PDFs need OCR first, which this pipeline doesn't include).

### 3.5 Build a ground-truth evaluation set

```bash
python -m scripts.generate_eval_set --n 100 --out data/eval/eval_qa_candidates.jsonl
```

This samples indexed chunks and asks the LLM to draft a question + answer for
each — **these are candidates, not ground truth**. Open the file, verify each
`reference_answer` against the actual source PDF, delete anything wrong, and
save the reviewed version as `data/eval/eval_qa.jsonl`. Skipping this step and
using unreviewed LLM-generated pairs as ground truth would let the same
model's blind spots grade its own homework — the resume claim of "100 verified
Q&A pairs" depends on the word "verified" actually being true.

### 3.6 Run the evaluation

```bash
python -m src.evaluation.run_eval --eval-set data/eval/eval_qa.jsonl \
    --index-dir data/index --out data/eval/report.json
```

Produces a JSON report comparing **no-RAG**, **naive-RAG** (single dense hit,
no reranking, no citation checking) and the **full pipeline** on:
retrieval Recall@5/10, Precision@5/10, MRR, nDCG; mean faithfulness; mean
correctness (LLM-as-judge); mean relevance; and mean latency per condition.
This is the artifact that turns "I built a RAG system" into "I built a RAG
system and proved it's better than the naive version, by X points of
faithfulness."

---

## 4. Running the system

### 4.1 Locally, without Docker

```bash
# Terminal 1 — backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
FINANCE_RAG_API_URL=http://localhost:8000 streamlit run frontend/app.py
```

Open http://localhost:8501. First backend startup will take a minute or two
while it loads the embedding, reranker, and NLI models (and the generation
model, if `LLM_BACKEND=hf_local`).

### 4.2 With Docker Compose

```bash
docker compose up --build
```

Starts the backend (port 8000), frontend (port 8501), and a Redis cache. The
backend mounts `data/index` read-only, so build the index on the host
**before** running Compose (§3.4) — the container doesn't build it for you.

### 4.3 API usage directly

```bash
curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What was TCS revenue in FY2024?"}'
```

Response shape (`api/schemas.py::QueryResponse`):

```json
{
  "query": "What was TCS revenue in FY2024?",
  "query_type": "factual",
  "answer": "According to the FY2024 annual report, TCS reported revenue of ...",
  "citations": [
    {
      "claim": "TCS reported revenue of Rs. X crore in FY2024.",
      "chunk_id": "…",
      "source_name": "TCS_Annual_Report_FY2024.pdf",
      "page_number": 42,
      "entailment_score": 0.94,
      "supported": true
    }
  ],
  "faithfulness_score": 1.0,
  "unsupported_claim_count": 0,
  "retrieved_chunk_ids": ["…", "…"],
  "latency_ms": 842.1,
  "cached": false
}
```

`/query/stream` emits the same result as Server-Sent Events with pipeline-
stage progress (`classifying` → `retrieving` → `reranking` → `generating` →
`verifying_citations` → `final`) so a UI can show progress instead of a bare
spinner during the ~1-3s round trip. See the docstring at the top of
`api/main.py` for why this streams *stages*, not raw LLM tokens: the
generator must emit one complete, valid JSON object for the citation
validator and hallucination checker to run against, so token-level streaming
of a partially-formed JSON blob isn't something this pipeline does safely.

Set `API_KEY` in `.env` to require an `X-API-Key` header on `/query*`; leave
it blank for local development.

### 4.4 Tests

```bash
pytest tests/ -v
```

Tests are designed to run **without any model weights or network access** —
they stub out `AppState.load` and heavy retrievers, and unit-test the RRF
fusion math, the token-aware chunker, and citation validation logic in
isolation. CI (`.github/workflows/ci.yml`) runs `ruff check` + `pytest` on
every push.

---

## 5. Deployment

- **Railway / Render / Fly.io**: point them at `Dockerfile.backend` for the
  API and `Dockerfile.frontend` for the UI as two services; set the same env
  vars from `.env`; either bake a pre-built `data/index/` into the backend
  image (simplest for a free-tier deploy with no persistent volume) or mount
  a volume and run `scripts/build_index.py` once against it.
- **HuggingFace Spaces**: works well for the Streamlit frontend alone if you
  point `FINANCE_RAG_API_URL` at a backend deployed elsewhere; Spaces' free
  CPU tier is generally too small to also host the backend's models.
- **GPU deployment**: swap `Dockerfile.backend`'s base image for a CUDA image
  and install the matching `torch` CUDA wheel instead of the CPU one; set
  `DEVICE=cuda`. This matters most if `LLM_BACKEND=hf_local` — the retrieval
  models (embedding/reranker/NLI) are small enough to run acceptably on CPU.

