# NyayaAI — Indian Legal Intelligence RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for Indian law. Ask questions about the Indian Constitution, Supreme Court judgments, Acts of Parliament, and Constituent Assembly Debates — and get structured, cited answers powered by AI.

![NyayaAI](https://img.shields.io/badge/NyayaAI-Indian%20Legal%20RAG-gold)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)

---

## What It Does

NyayaAI answers legal queries by:
1. Embedding your question using a local GPU-accelerated model
2. Searching 30,000+ vector chunks from SC judgments, Acts, Constitution, and CAD debates
3. Feeding the most relevant passages to an LLM (Claude/DeepSeek/Llama)
4. Returning a structured legal answer with citations and source scores

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
├── api/
│   └── main.py              # FastAPI backend
├── embeddings/
│   └── batch_ingest.py      # GPU embedding + Qdrant upload
├── frontend/
│   └── index.html           # NyayaAI chat UI
├── ingestion/
│   ├── scraper/
│   │   ├── commonLIT_scraper.py   # SC judgments (AWS S3)
│   │   ├── Devgan_scraper.py      # Acts section-by-section
│   │   ├── Legislative_scraper.py # Constitution + Acts PDFs
│   │   └── CAD_scaper.py          # Constituent Assembly Debates
│   ├── cleaner/
│   │   └── PDF_text_extracter.py  # PDF → JSON text extraction
│   └── run_all_scraper.py         # Master scraper runner
├── retrieval/               # (Future: reranker, query classifier)
├── data/                    # Scraped data (gitignored)
├── .env.example             # Environment variables template
├── requirements.txt
└── README.md
```

---

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
git clone https://github.com/YOUR_USERNAME/Indian-legal-RAG.git
cd Indian-legal-RAG
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
  "model_used": "anthropic/claude-sonnet-4-5"
}
```

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
User Query
    │
    ▼
Embedding Model (all-MiniLM-L6-v2 on GPU)
    │
    ▼
Qdrant Vector Search (cosine similarity, top-8 chunks)
    │
    ▼
Context Assembly (judgment + act + CAD chunks)
    │
    ▼
LLM Generation (Claude/DeepSeek via OpenRouter)
    │
    ▼
Structured Legal Answer + Cited Sources
```

---

## Roadmap

- [ ] Reranker (Cohere) for better result ordering
- [ ] Query classifier (route to judgment/act/CAD automatically)
- [ ] More SC judgments (scale to 50,000+)
- [ ] High Court judgments
- [ ] Multilingual support (Hindi, Tamil, Marathi)
- [ ] Deploy to cloud (Railway / Render / AWS)

---

## Legal Disclaimer

NyayaAI provides legal **information**, not legal **advice**. Always consult a qualified lawyer for personal legal matters. The system may not reflect the most recent legal developments.

---

## License

MIT License — free to use, modify, and distribute.
