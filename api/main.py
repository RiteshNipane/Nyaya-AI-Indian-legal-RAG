"""
api/main.py

FastAPI backend for Indian Legal RAG.
Query → Retrieve from Qdrant → Generate answer via OpenRouter → Return structured response.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
import torch
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "indian_legal_rag"
TOP_K = 8

# ─────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading embedding model on {DEVICE.upper()}...")
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)
print("✓ Embedding model loaded")

qdrant = QdrantClient(url=QDRANT_URL)

# Correctly configured for OpenRouter
llm_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0
)

app = FastAPI(title="Indian Legal RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    doc_type: Optional[str] = None
    model: str = "anthropic/claude-3.5-sonnet"  # Valid OpenRouter Default


class SourceDoc(BaseModel):
    doc_type: str
    text_snippet: str
    title: str = ""
    source: str = ""
    score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    query: str
    model_used: str

# ─────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────


def retrieve(query: str, top_k: int = TOP_K, doc_type: str = None) -> list:
    """Embed query and retrieve top_k chunks from Qdrant."""

    query_vector = embed_model.encode(
        [query],
        normalize_embeddings=True,
        device=DEVICE
    )[0].tolist()

    query_filter = None
    if doc_type:
        query_filter = Filter(
            must=[FieldCondition(
                key="doc_type",
                match=MatchValue(value=doc_type)
            )]
        )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True
    ).points

    return results


def format_source_label(payload: dict) -> str:
    """Build a readable source label from chunk metadata."""
    doc_type = payload.get("doc_type", "")

    if doc_type == "judgment":
        title = payload.get("title", "Unknown Case")
        year = payload.get("year", "")
        court = payload.get("court", "Supreme Court of India")
        return f"{title} ({year}) — {court}"

    elif doc_type == "act":
        act_name = payload.get("act_name") or payload.get("act", "").upper()
        section = payload.get("section_number", "")
        title = payload.get("section_title", "")
        return f"{act_name}, Section {section} — {title}"

    elif doc_type == "cad":
        volume = payload.get("volume", "")
        date = payload.get("date", "")
        speaker = payload.get("speaker", "")
        return f"CAD Vol.{volume} ({date}) — {speaker}"

    return payload.get("source", "Legal Source")

# ─────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are an expert Indian constitutional lawyer and legal researcher.

You help lawyers, students, and citizens understand Indian law by providing:
1. Relevant constitutional articles and their interpretation
2. Landmark Supreme Court and High Court judgments
3. The legal reasoning behind key principles
4. Framer intent from Constituent Assembly Debates when relevant

STRICT RULES:
- ONLY use information from the provided context documents
- If context is insufficient, say so clearly — never hallucinate cases or articles
- Always cite: case name + year + court for every judgment
- Always cite: Article number for every constitutional provision
- Distinguish binding precedent (SC) from persuasive authority (HC)
- Never give direct legal advice — provide legal information only

OUTPUT FORMAT:
1. **Relevant Constitutional Provisions** — Articles that apply
2. **Key Judgments** — Case, year, court, and what was held
3. **Legal Reasoning** — The principle flowing from these sources
4. **Framer Intent** (if available from CAD) — Original intent
5. **Caveats** — Limitations or conflicting views
"""


def build_prompt(query: str, results: list) -> str:
    context_parts = []
    for i, result in enumerate(results, 1):
        payload = result.payload
        label = format_source_label(payload)
        text = payload.get("text", "")
        context_parts.append(f"[Source {i}: {label}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""Answer this legal query using ONLY the sources provided below.

QUERY: {query}

LEGAL SOURCES:
{context}

Provide a structured answer. Cite sources as [Source N]."""


def generate_answer(query: str, results: list, model: str) -> str:
    prompt = build_prompt(query, results)

    try:
        response = llm_client.chat.completions.create(
            model=model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        # This prevents silent crashes and prints the exact OpenRouter error to your terminal
        print(f"\n❌ OPENROUTER API ERROR: {e}\n")
        raise HTTPException(
            status_code=502, detail=f"LLM Generation failed: {str(e)}")

# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "ok", "message": "Indian Legal RAG API is running"}


@app.get("/health")
def health():
    try:
        count = qdrant.count(COLLECTION_NAME).count
        return {
            "status":  "ok",
            "qdrant":  "connected",
            "vectors": count,
            "device":  DEVICE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Retrieve
    results = retrieve(
        query=request.query,
        top_k=request.top_k,
        doc_type=request.doc_type
    )

    if not results:
        raise HTTPException(
            status_code=404, detail="No relevant documents found")

    # Generate
    answer = generate_answer(request.query, results, request.model)

    # Format sources
    sources = [
        SourceDoc(
            doc_type=r.payload.get("doc_type", ""),
            text_snippet=r.payload.get("text", "")[:200] + "...",
            title=format_source_label(r.payload),
            source=r.payload.get("source", ""),
            score=round(r.score, 4)
        )
        for r in results
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        query=request.query,
        model_used=request.model
    )


@app.get("/collections")
def collection_info():
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        count = qdrant.count(COLLECTION_NAME).count
        return {
            "name":    COLLECTION_NAME,
            "vectors": count,
            "status":  str(info.status)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
