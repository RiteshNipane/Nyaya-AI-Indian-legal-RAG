"""
retrieval/adapters.py

Concrete adapters for the retrieval pipeline: sentence-transformers embedder,
Qdrant vector store, and a factory that wires everything from environment
variables. Heavy imports (torch, sentence-transformers, qdrant-client) are
deferred so the rest of the `retrieval` package stays importable without them.

Environment
-----------
QDRANT_URL, QDRANT_API_KEY      Qdrant location (API key needed for Qdrant Cloud)
EMBEDDING_MODEL                 default sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_QUERY_PREFIX          e.g. "query: " for E5 models (default empty)
RERANKER                        auto | cohere | cross-encoder | bm25 | none
CANDIDATE_K                     dense pool size fed to the reranker (default 30)
OPENROUTER_API_KEY              LLM generation + query translation
TRANSLATION_MODEL               default anthropic/claude-3.5-haiku
"""

from __future__ import annotations

import os
from typing import Optional

from retrieval.multilingual import LLMTranslator
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_classifier import QueryClassifier
from retrieval.reranker import get_reranker
from retrieval.types import Candidate

DEFAULT_COLLECTION = "indian_legal_rag"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 query_prefix: Optional[str] = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_prefix = (
            query_prefix if query_prefix is not None
            else os.getenv("EMBEDDING_QUERY_PREFIX", "")
        )
        self.model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_query(self, text: str) -> list:
        return self.model.encode(
            [self.query_prefix + text], normalize_embeddings=True, device=self.device
        )[0].tolist()


class QdrantStore:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None,
                 collection: str = DEFAULT_COLLECTION, client=None):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = client or QdrantClient(
            url=url or os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key or os.getenv("QDRANT_API_KEY") or None,
        )

    @staticmethod
    def _filter(doc_types: Optional[list]):
        if not doc_types:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        if len(doc_types) == 1:
            cond = FieldCondition(key="doc_type", match=MatchValue(value=doc_types[0]))
        else:
            cond = FieldCondition(key="doc_type", match=MatchAny(any=list(doc_types)))
        return Filter(must=[cond])

    def search(self, vector: list, limit: int, doc_types: Optional[list] = None) -> list:
        points = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=self._filter(doc_types),
            limit=limit,
            with_payload=True,
        ).points
        return [Candidate(id=str(p.id), payload=p.payload or {}, score=float(p.score))
                for p in points]

    def count(self) -> int:
        return self.client.count(self.collection).count

    def vector_size(self) -> Optional[int]:
        try:
            cfg = self.client.get_collection(self.collection).config.params.vectors
            return getattr(cfg, "size", None)
        except Exception:
            return None


def build_llm_client():
    """OpenAI-compatible client pointed at OpenRouter (or None without a key)."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=60.0)


def build_pipeline_from_env(llm_client=None) -> tuple:
    """Return (pipeline, store, embedder, llm_client) built from the environment."""
    embedder = SentenceTransformerEmbedder()
    store = QdrantStore()
    llm_client = llm_client or build_llm_client()

    translator = None
    if llm_client is not None:
        translator = LLMTranslator(
            llm_client, os.getenv("TRANSLATION_MODEL", "anthropic/claude-3.5-haiku"))

    pipeline = RetrievalPipeline(
        embedder=embedder,
        store=store,
        reranker=get_reranker(),
        classifier=QueryClassifier(),
        translator=translator,
        candidate_k=int(os.getenv("CANDIDATE_K", "30")),
    )
    return pipeline, store, embedder, llm_client
