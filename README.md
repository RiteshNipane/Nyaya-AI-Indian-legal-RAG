# NyayaAI — Indian Legal Intelligence RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for Indian law. Ask questions about the Indian Constitution, Supreme Court judgments, Acts of Parliament, and Constituent Assembly Debates — and get structured, cited answers powered by AI.

![NyayaAI](https://img.shields.io/badge/NyayaAI-Indian%20Legal%20RAG-gold)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)

---

## What It Does

NyayaAI answers legal queries by:
1. Detecting the question's language and translating non-English queries (Hindi, Marathi, Tamil, ...) for retrieval
2. Classifying the question and routing it to acts, judgments or Constituent Assembly Debates
3. Embedding it with a local GPU-accelerated model and retrieving a wide candidate pool from Qdrant
4. Reranking the candidates (Cohere, a local cross-encoder, or BM25 as an offline fallback)
5. Feeding the best passages to an LLM (Claude/DeepSeek/Llama) and answering in the user's language
6. Returning a structured legal answer with citations, dense and rerank scores

---

## Data Sources

| Source | Content | Volume |
|--------|---------|--------|
| AWS Open Data | Supreme Court judgments 1950–2025 | 150+ judgments |
| Devgan.in | BNS, BNSS, BSA, IPC, CrPC sections | 1,500+ sections |
| legislative.gov.in | Constitution of India, RTI, IT Act, POCSO | 8 Acts |
| constitutionofindia.net | Constituent Assembly Debates (12 volumes) | 60+ sessions |

---

## Project Structure

```
Indian-legal-RAG/
├── api/main.py                    # FastAPI backend (thin; logic lives in retrieval/)
├── retrieval/
│   ├── query_classifier.py        # rule-based routing: act / judgment / cad
│   ├── reranker.py                # Cohere, cross-encoder, BM25 + fallback chain
│   ├── pipeline.py                # translate → classify → retrieve → rerank
│   ├── multilingual.py            # script detection + LLM query translation
│   ├── generation.py              # prompt building + answer generation
│   ├── adapters.py                # Qdrant / sentence-transformers / OpenRouter wiring
│   └── types.py                   # shared types, source labels
├── evaluation/
│   ├── metrics.py                 # hit@k, P@k, pooled recall@k, MRR, nDCG@k
│   ├── generation_eval.py         # citation checks, keyword coverage, LLM judge
│   ├── run_eval.py                # ablation runner + JSON/Markdown reports
│   └── golden_set.jsonl           # 28 starter queries (statute/constitution/case law/CAD/multilingual)
├── embeddings/batch_ingestion.py  # GPU embedding + Qdrant upload (idempotent)
├── ingestion/
│   ├── scraper/                   # SC (streaming, parallel), High Courts, Devgan, Legislative, CAD
│   ├── cleaner/PDF_text_extracter.py
│   └── run_all_scraper.py         # mvp | full | scale
├── frontend/index.html            # chat UI (API base configurable)
├── tests/                         # pytest suite (no GPU / Qdrant / network needed)
├── Dockerfile, docker-compose.yml, render.yaml, railway.json
├── .github/workflows/ci.yml
├── .env.example
└── requirements.txt, requirements-api.txt
```

* * *

## Prerequisites

- **Python 3.12.x** (not 3.13 or 3.14 — PyTorch requires 3.12)
  - Download: https://python.org/downloads/release/python-31210/
- **Docker Desktop** (for Qdrant vector database)
  - Download: https://docker.com/products/docker-desktop
- **NVIDIA GPU** with CUDA support (optional but recommended)
  - CPU works but is much slower for embedding
- **OpenRouter API key** (for LLM generation)
  - Get one free at: https://openrouter.ai/keys

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/RiteshNipane/Nyaya-AI-Indian-legal-RAG.git
cd Nyaya-AI-Indian-legal-RAG
```

### 2. Install Python dependencies

```bash
# Install PyTorch with CUDA (for GPU acceleration)
py -3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
py -3.12 -m pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your keys
```

`.env` file:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
QDRANT_URL=http://localhost:6333
```

### 4. Start Qdrant (vector database)

```bash
docker run -p 6333:6333 -p 6334:6334 -v "%USERPROFILE%\qdrant_storage:/qdrant/storage" qdrant/qdrant
```

---

## Data Collection (One-time Setup)

Run the scrapers to collect legal data:

```bash
# Run all scrapers (takes 30-60 minutes)
py -3.12 ingestion/run_all_scraper.py
```

Or run individually:
```bash
# SC Judgments from AWS Open Data
py -3.12 ingestion/scraper/commonLIT_scraper.py

# Act sections from Devgan.in
py -3.12 ingestion/scraper/Devgan_scraper.py

# Constitution + Acts PDFs
py -3.12 ingestion/scraper/Legislative_scraper.py

# Extract text from downloaded PDFs
py -3.12 ingestion/cleaner/PDF_text_extracter.py

# Constituent Assembly Debates
py -3.12 ingestion/scraper/CAD_scaper.py

# High Court judgments (sample two courts first)
py -3.12 ingestion/scraper/highcourt_scraper.py --courts 27_1 7_26 --years 2023-2024 --max-per-court-year 300
py -3.12 ingestion/scraper/highcourt_scraper.py --list-courts
```

---

## Embedding & Ingestion

Embed all scraped data and store in Qdrant:

```bash
py -3.12 embeddings/batch_ingest.py
```

This will:
- Load `sentence-transformers/all-MiniLM-L6-v2` on your GPU
- Chunk all JSON files into 600-char segments
- Embed using GPU (RTX 3050: ~1.4 minutes for 220 files)
- Upload 15,000+ vectors to Qdrant with metadata filters

---

## Running the Application

### Start the API

```bash
py -3.12 api/main.py
```

API runs at `http://localhost:8000`

- `GET /health` — Check status and vector count
- `POST /query` — Query the RAG system
- `GET /docs` — Interactive Swagger UI

### Open the Frontend

Open `frontend/index.html` in your browser. No server needed.

---

## Usage Example

Query via Swagger UI at `http://localhost:8000/docs` or directly:

```json
POST /query
{
  "query": "What is the punishment for murder under Indian law?",
  "top_k": 8
}
```

Response:
```json
{
  "answer": "**1. Relevant Constitutional Provisions**...",
  "sources": [
    {
      "doc_type": "act",
      "title": "BNS, Section 101 — Punishment for murder",
      "score": 0.8923
    }
  ],
  "model_used": "anthropic/claude-sonnet-4-5",
  "language": "en",
  "query_type": "statute",
  "routed_doc_types": ["act"],
  "reranker": "cohere",
  "timings_ms": {"retrieve": 41, "rerank": 230}
}
```

Optional request fields: `rerank` (default true), `route` (default true), `language` (auto-detected), `doc_type`.
`GET /classify?q=...` shows how a query would be routed.

---

## Dependencies

```
fastapi
uvicorn
qdrant-client
sentence-transformers
torch
torchvision
torchaudio
openai
python-dotenv
requests
beautifulsoup4
lxml
PyPDF2
tqdm
pydantic
```

---

## Architecture

```
User Query (any supported language)
    │
    ▼
Language detection ─► LLM translation to English (non-English only)
    │
    ▼
Query classifier ─► doc_type routing (act / judgment / cad), widens if results are thin
    │
    ▼
Embedding model (all-MiniLM-L6-v2, configurable) ─► Qdrant, top CANDIDATE_K (30)
    │
    ▼
Reranker (Cohere → local cross-encoder → BM25 fallback) ─► top_k (8)
    │
    ▼
LLM generation via OpenRouter (answers in the user's language)
    │
    ▼
Structured Legal Answer + Cited Sources
```

### Reranker

`RERANKER=auto` (default) tries Cohere when `COHERE_API_KEY` is set, then a local
`sentence-transformers` cross-encoder, then a pure-Python BM25 scorer. If a reranker
errors or times out the next one in the chain is used, so `/query` never fails because
of reranking. Force one with `RERANKER=cohere|cross-encoder|bm25|none`.

### Query routing

The classifier is rule-based (no model call). It only restricts the search when the
evidence is strong (e.g. "section 103 BNS", "X v. Y", "what did Ambedkar say"); mixed or
weak signals search everything, and a routed search that returns too few chunks is
widened automatically. Plain constitutional questions deliberately search the whole corpus.

### Multilingual

The English index is kept. Non-English queries are detected by Unicode script
(Devanagari is split into Hindi/Marathi by common words), translated to English for
retrieval, and the answer is generated in the user's language while statute names,
section numbers and case names stay in English. Set `TRANSLATION_MODEL` to pick the
translation model. To go fully multilingual instead, set `EMBEDDING_MODEL` (e.g.
`BAAI/bge-m3`) plus the query/doc prefixes and **re-ingest**.

* * *

## Evaluation

```bash
# Offline: routing accuracy of the query classifier (no Qdrant or models needed)
python -m evaluation.run_eval --classifier-only

# Retrieval ablation: baseline vs rerank vs route vs rerank+route (needs Qdrant + ingested data)
python -m evaluation.run_eval --ks 1 3 5 10 --candidate-k 30

# Add answer quality: citation validity, keyword coverage, and LLM-judge faithfulness/relevance
python -m evaluation.run_eval --generation --judge
```

Reports (JSON + Markdown) are written to `evaluation/results/`.

Metrics: hit@k, precision@k, **pooled** recall@k, MRR, nDCG@k, pool coverage and latency
percentiles per variant; for answers: citation rate/validity, context coverage, keyword
coverage, and optional LLM-judge scores.

The index has no hand-labelled chunk IDs, so each golden query lists *matchers*
(`doc_type`, `act`, `section_number`, `title_contains`, `text_contains_any`, ...) and a
chunk counts as relevant if it matches any of them. Recall and nDCG are pooled against the
relevant chunks found in the dense candidate pool, which is what a reranker can change.
`evaluation/golden_set.jsonl` is a starter set: **extend it with queries and matchers for
the corpus you actually ingested** before trusting absolute numbers.

* * *

## Scaling

* **50,000+ SC judgments:** `python ingestion/run_all_scraper.py scale` streams each year's
  tar (no full-archive-in-RAM), extracts PDFs in parallel, and resumes safely. Or call
  `commonLIT_scraper.py --years 1950-2025 --max-per-year 0`.
* **Ingest at scale:** `QDRANT_ON_DISK=1 QDRANT_QUANTIZE=1 py embeddings/batch_ingestion.py`
  keeps vectors on disk with int8 quantisation. Point IDs are deterministic, so re-running
  ingestion overwrites instead of duplicating.
* **High Courts:** see `highcourt_scraper.py`; chunks are stored as `doc_type: judgment`
  with the High Court name in `court`.

* * *

## Deployment

The API ships as a Docker image (`Dockerfile`, CPU PyTorch, embedding model baked in).

1. Create a Qdrant Cloud cluster and ingest into it:
   `QDRANT_URL=https://<cluster> QDRANT_API_KEY=... py embeddings/batch_ingestion.py`
2. **Render:** connect the repo; `render.yaml` defines the service. Set
   `OPENROUTER_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` (and optionally `COHERE_API_KEY`).
3. **Railway:** `railway.json` uses the Dockerfile; set the same variables.
4. **Local full stack:** `docker compose up --build` (API + Qdrant).
5. Host `frontend/index.html` anywhere static and open it with `?api=https://your-api-host`
   (or set `window.NYAYA_API_URL`). Set `CORS_ORIGINS` to your frontend's origin in production.

* * *

## Tests

```bash
python -m pytest -q
```

The suite uses fakes, so it needs no GPU, Qdrant or network. CI runs it on every push.

* * *

## Roadmap

- [x] Reranker (Cohere, local cross-encoder, BM25 fallback)
- [x] Query classifier (route to judgment/act/CAD automatically)
- [x] Evaluation metrics and ablation harness
- [x] More SC judgments (streaming, parallel, resumable pipeline for 50,000+)
- [x] High Court judgments
- [x] Multilingual support (Hindi, Marathi, Tamil + other Indian scripts via translation)
- [x] Deploy to cloud (Docker, Render, Railway, docker-compose)

* * *

## Legal Disclaimer

NyayaAI provides legal **information**, not legal **advice**. Always consult a qualified lawyer for personal legal matters. The system may not reflect the most recent legal developments.

---

## License

MIT License — free to use, modify, and distribute.
