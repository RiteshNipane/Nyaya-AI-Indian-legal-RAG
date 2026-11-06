"""
api/main.py

FastAPI backend for Indian Legal RAG.

Query → (translate) → classify/route → dense retrieve → rerank → generate → respond.
"""

import os
import sys
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from retrieval.adapters import build_pipeline_from_env  # noqa: E402
from retrieval.generation import GenerationError, generate_answer  # noqa: E402
from retrieval.query_classifier import QueryClassifier  # noqa: E402
from retrieval.types import format_source_label  # noqa: E402

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

TOP_K = int(os.getenv("TOP_K", "8"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-3.5-sonnet")
DOC_TYPES = {"act", "constitution", "judgment", "cad"}

# ─────────────────────────────────────────────────────────
# CLIENTS  (embedder, Qdrant, reranker, translator, LLM)
# ─────────────────────────────────────────────────────────

print("Initialising retrieval pipeline...")
pipeline, store, embedder, llm_client = build_pipeline_from_env()
print(f"✓ Embedding model loaded on {embedder.device.upper()}")
print(f"✓ Reranker: {pipeline.reranker.name if pipeline.reranker else 'disabled'}")

_collection_dim = store.vector_size()
if _collection_dim and _collection_dim != embedder.dimension:
    print(
        f"⚠ Embedding dimension ({embedder.dimension}) does not match the Qdrant "
        f"collection ({_collection_dim}). Re-ingest or fix EMBEDDING_MODEL."
    )

app = FastAPI(title="Indian Legal RAG API", version="2.0")

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # credentials are not used; wildcard origins and credentials cannot be combined
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=TOP_K, ge=1, le=30)
    doc_type: Optional[str] = None
    model: str = DEFAULT_MODEL
    rerank: bool = True      # second-stage reranking
    route: bool = True       # automatic act/judgment/CAD routing
    language: Optional[str] = None   # ISO code (hi, mr, ta, ...); auto-detected if omitted


class SourceDoc(BaseModel):
    doc_type: str
    text_snippet: str
    title: str = ""
    source: str = ""
    score: float = 0.0                    # final score (reranker score when reranked)
    dense_score: float = 0.0
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    query: str
    model_used: str
    language: str = "en"
    retrieval_query: str = ""
    query_type: str = "general"
    routed_doc_types: Optional[list[str]] = None
    widened: bool = False
    reranker: Optional[str] = None
    timings_ms: dict = {}


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "ok", "message": "Indian Legal RAG API is running"}


@app.get("/health")
def health():
    try:
        return {
            "status": "ok",
            "qdrant": "connected",
            "vectors": store.count(),
            "device": embedder.device,
            "reranker": pipeline.reranker.name if pipeline.reranker else None,
            "translation": pipeline.translator is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classify")
def classify(q: str = Query(..., min_length=1)):
    """Debug endpoint: show how a query would be routed."""
    return QueryClassifier().classify(q).to_dict()


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if request.doc_type and request.doc_type not in DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"doc_type must be one of {sorted(DOC_TYPES)}")

    try:
        result = pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            doc_type=request.doc_type,
            rerank=request.rerank,
            route=request.route,
            language=request.language,
        )
    except Exception as e:
        print(f"\n❌ RETRIEVAL ERROR: {e}\n")
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {e}")

    if not result.candidates:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    try:
        answer = generate_answer(
            llm_client, request.model, request.query,
            result.candidates, result.language)
    except GenerationError as e:
        print(f"\n❌ OPENROUTER API ERROR: {e}\n")
        raise HTTPException(status_code=502, detail=f"LLM Generation failed: {e}")

    sources = [
        SourceDoc(
            doc_type=c.payload.get("doc_type", ""),
            text_snippet=c.text[:200] + "...",
            title=format_source_label(c.payload),
            source=c.payload.get("source", ""),
            score=round(c.final_score, 4),
            dense_score=round(c.score, 4),
            rerank_score=None if c.rerank_score is None else round(c.rerank_score, 4),
        )
        for c in result.candidates
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        query=request.query,
        model_used=request.model,
        language=result.language.code,
        retrieval_query=result.retrieval_query,
        query_type=result.classification.query_type if result.classification else "general",
        routed_doc_types=result.routed_doc_types,
        widened=result.widened,
        reranker=result.reranker,
        timings_ms=result.timings_ms,
    )


@app.get("/collections")
def collection_info():
    try:
        info = store.client.get_collection(store.collection)
        return {
            "name": store.collection,
            "vectors": store.count(),
            "status": str(info.status),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "1") == "1",
    )
