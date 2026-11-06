"""
retrieval/reranker.py

Second-stage rerankers. The dense retriever (all-MiniLM-L6-v2 + Qdrant) is
fast but coarse; a reranker re-scores the top-N candidates against the query
jointly, which noticeably improves ordering for legal text.

Backends
--------
* CohereReranker        – Cohere Rerank API over plain HTTPS (needs COHERE_API_KEY).
* CrossEncoderReranker  – local sentence-transformers CrossEncoder (free, GPU/CPU).
* BM25Reranker          – dependency-free lexical fallback.
* FallbackReranker      – tries backends in order so the API never breaks.

Select with env RERANKER = auto | cohere | cross-encoder | bm25 | none
(default "auto": Cohere if COHERE_API_KEY is set, else cross-encoder if
sentence-transformers is installed, else BM25).
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import re
import time
from dataclasses import replace
from typing import Optional, Protocol

from retrieval.types import Candidate, format_source_label

logger = logging.getLogger("nyaya.reranker")


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list, top_k: int) -> list:
        ...


def candidate_document(c: Candidate) -> str:
    """Text shown to the reranker: source label + chunk text.

    Prepending the label (case name, act + section) helps the model match
    queries like "BNS section 103" that name the source rather than the content.
    """
    return f"{format_source_label(c.payload)}\n{c.text}"


# ─────────────────────────────────────────────────────────
# Cohere
# ─────────────────────────────────────────────────────────

class CohereReranker:
    name = "cohere"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: float = 15.0, max_retries: int = 2,
                 url: str = "https://api.cohere.com/v2/rerank", session=None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not set")
        self.model = model or os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
        self.timeout = timeout
        self.max_retries = max_retries
        self.url = url
        self._session = session

    def _post(self, body: dict) -> dict:
        import requests  # local import keeps module importable without requests

        session = self._session or requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = session.post(self.url, json=body, headers=headers,
                                    timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"Cohere HTTP {resp.status_code}")
                    time.sleep(min(2 ** attempt, 4))
                    continue
                raise RuntimeError(
                    f"Cohere HTTP {resp.status_code}: {resp.text[:200]}")
            except RuntimeError:
                raise
            except Exception as e:  # network errors, timeouts
                last_err = e
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"Cohere rerank failed: {last_err}")

    def rerank(self, query: str, candidates: list, top_k: int) -> list:
        if not candidates:
            return []
        body = {
            "model": self.model,
            "query": query,
            "documents": [candidate_document(c) for c in candidates],
            "top_n": min(top_k, len(candidates)),
        }
        data = self._post(body)
        results = data.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Unexpected Cohere response: missing 'results'")
        out = []
        for item in results:
            idx = item["index"]
            out.append(replace(candidates[idx],
                               rerank_score=float(item["relevance_score"])))
        out.sort(key=lambda c: c.rerank_score, reverse=True)
        return out[:top_k]


# ─────────────────────────────────────────────────────────
# Local cross-encoder
# ─────────────────────────────────────────────────────────

class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 batch_size: int = 32, max_length: int = 512):
        self.model_name = model_name or os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            kwargs = {"max_length": self.max_length}
            if self.device:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(self, query: str, candidates: list, top_k: int) -> list:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, candidate_document(c)) for c in candidates]
        raw = model.predict(pairs, batch_size=self.batch_size,
                            show_progress_bar=False)
        scored = []
        for c, s in zip(candidates, raw):
            s = float(s)
            prob = 1.0 / (1.0 + math.exp(-s))  # logits -> (0, 1)
            scored.append(replace(c, rerank_score=prob))
        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        return scored[:top_k]


# ─────────────────────────────────────────────────────────
# BM25 (no dependencies)
# ─────────────────────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have how in is it of on or that the "
    "this to was were what when which who why will with under can does do did".split()
)


def _tokens(text: str) -> list:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Reranker:
    """Okapi BM25 computed over the candidate set only (cheap, offline)."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def rerank(self, query: str, candidates: list, top_k: int) -> list:
        if not candidates:
            return []
        q_terms = _tokens(query)
        docs = [_tokens(candidate_document(c)) for c in candidates]
        n = len(docs)
        avgdl = (sum(len(d) for d in docs) / n) or 1.0

        df: dict = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1

        scores = []
        for d in docs:
            tf: dict = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q_terms:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                denom = tf[t] + self.k1 * (1 - self.b + self.b * len(d) / avgdl)
                s += idf * tf[t] * (self.k1 + 1) / denom
            scores.append(s)

        top = max(scores) or 1.0
        scored = [
            replace(c, rerank_score=s / top)  # normalise to 0..1
            for c, s in zip(candidates, scores)
        ]
        # stable: ties keep the dense order
        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        return scored[:top_k]


# ─────────────────────────────────────────────────────────
# Fallback chain
# ─────────────────────────────────────────────────────────

class FallbackReranker:
    """Try rerankers in order; if all fail, keep the dense ordering."""

    def __init__(self, rerankers: list):
        if not rerankers:
            raise ValueError("FallbackReranker needs at least one reranker")
        self.rerankers = rerankers
        self.last_used: Optional[str] = None

    @property
    def name(self) -> str:
        return "+".join(r.name for r in self.rerankers)

    def rerank(self, query: str, candidates: list, top_k: int) -> list:
        for r in self.rerankers:
            try:
                out = r.rerank(query, candidates, top_k)
                self.last_used = r.name
                return out
            except Exception as e:
                logger.warning("Reranker '%s' failed (%s); trying next", r.name, e)
        self.last_used = "none"
        return candidates[:top_k]


# ─────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────

def get_reranker(kind: Optional[str] = None) -> Optional[Reranker]:
    """Build a reranker from `kind` or env RERANKER. Returns None for 'none'."""
    kind = (kind or os.getenv("RERANKER", "auto")).strip().lower()

    if kind in ("none", "off", "false", "0"):
        return None
    if kind == "bm25":
        return BM25Reranker()
    if kind == "cohere":
        return FallbackReranker([CohereReranker(), BM25Reranker()])
    if kind in ("cross-encoder", "cross_encoder", "local"):
        return FallbackReranker([CrossEncoderReranker(), BM25Reranker()])
    if kind != "auto":
        raise ValueError(f"Unknown RERANKER '{kind}'")

    chain: list = []
    if os.getenv("COHERE_API_KEY"):
        chain.append(CohereReranker())
    if importlib.util.find_spec("sentence_transformers") is not None:
        chain.append(CrossEncoderReranker())
    chain.append(BM25Reranker())
    return FallbackReranker(chain)
