"""
retrieval/pipeline.py

Retrieval pipeline:

    query ─► language detect/translate ─► classify/route ─► dense search (N candidates)
          ─► (widen if routed search was too thin) ─► rerank ─► top_k

Dependencies (embedder, vector store, reranker, classifier, translator) are
injected, so the pipeline is unit-testable and the evaluation harness can reuse
exactly the code path the API uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from retrieval.multilingual import LanguageInfo, prepare_query
from retrieval.query_classifier import QueryClassification, QueryClassifier
from retrieval.types import Candidate


class Embedder(Protocol):
    def encode_query(self, text: str) -> list:
        ...


class VectorStore(Protocol):
    def search(self, vector: list, limit: int, doc_types: Optional[list] = None) -> list:
        ...


@dataclass
class RetrievalResult:
    candidates: list                      # final top_k, best first
    pool: list                            # dense candidate pool (pre-rerank)
    retrieval_query: str
    language: LanguageInfo
    classification: Optional[QueryClassification] = None
    routed_doc_types: Optional[list] = None
    widened: bool = False
    reranker: Optional[str] = None
    timings_ms: dict = field(default_factory=dict)


def _dedupe(candidates: list) -> list:
    seen: set = set()
    out = []
    for c in candidates:
        if c.id in seen:
            continue
        seen.add(c.id)
        out.append(c)
    return out


class RetrievalPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore,
                 reranker=None, classifier: Optional[QueryClassifier] = None,
                 translator=None, candidate_k: int = 30, min_results: int = 3):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.classifier = classifier
        self.translator = translator
        self.candidate_k = candidate_k
        self.min_results = min_results

    def retrieve(self, query: str, top_k: int = 8, doc_type: Optional[str] = None,
                 rerank: bool = True, route: bool = True,
                 language: Optional[str] = None) -> RetrievalResult:
        timings: dict = {}
        t0 = time.perf_counter()

        # 1. language + translation
        retrieval_query, lang = prepare_query(query, self.translator, language)
        timings["translate"] = (time.perf_counter() - t0) * 1000

        # 2. routing: an explicit doc_type from the client always wins
        classification = None
        routed: Optional[list] = None
        t = time.perf_counter()
        if doc_type:
            routed = [doc_type]
        elif route and self.classifier is not None:
            classification = self.classifier.classify(retrieval_query)
            routed = classification.doc_types
        timings["classify"] = (time.perf_counter() - t) * 1000

        # 3. dense search for a wide candidate pool
        pool_size = max(self.candidate_k, top_k) if (rerank and self.reranker) else top_k
        t = time.perf_counter()
        vector = self.embedder.encode_query(retrieval_query)
        timings["embed"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        pool = self.store.search(vector, pool_size, routed)

        # 4. widen to the full corpus when a routed search is too thin
        widened = False
        if routed and not doc_type and len(pool) < min(self.min_results, top_k):
            extra = self.store.search(vector, pool_size, None)
            pool = _dedupe(pool + extra)
            pool.sort(key=lambda c: c.score, reverse=True)
            pool = pool[:pool_size]
            widened = True
        timings["search"] = (time.perf_counter() - t) * 1000

        # 5. rerank
        reranker_name = None
        t = time.perf_counter()
        if rerank and self.reranker is not None and pool:
            final = self.reranker.rerank(retrieval_query, pool, top_k)
            reranker_name = getattr(self.reranker, "last_used", None) or self.reranker.name
        else:
            final = pool[:top_k]
        timings["rerank"] = (time.perf_counter() - t) * 1000
        timings["total"] = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            candidates=final, pool=pool, retrieval_query=retrieval_query,
            language=lang, classification=classification, routed_doc_types=routed,
            widened=widened, reranker=reranker_name,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
        )
