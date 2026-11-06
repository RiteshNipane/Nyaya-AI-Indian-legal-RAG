"""
evaluation/metrics.py

Retrieval metrics. Pure Python, no third-party dependencies.

Relevance without document IDs
------------------------------
The index has no stable, hand-labelled IDs, so a golden query lists *matchers*
and a retrieved chunk is relevant if it satisfies ANY matcher. A matcher is a
dict whose keys must ALL hold (AND):

    doc_type            exact match on payload["doc_type"]
    act                 payload["act"] equals it, or it is a substring of payload["act_name"]
    section_number      int/str or list of them (compared as strings, case-insensitive)
    title_contains      substring(s) of payload title / section_title
    speaker_contains    substring(s) of payload["speaker"]
    text_contains_any   at least one phrase occurs in the chunk text
    text_contains_all   every phrase occurs in the chunk text

All string comparisons are case-insensitive.

Pooled recall / nDCG
--------------------
The true number of relevant chunks in the corpus is unknown, so recall@k and the
ideal DCG are *pooled*: computed against the relevant chunks found in the dense
candidate pool (top-N before reranking). Recall@k therefore answers "of the
relevant chunks the retriever surfaced at all, how many made the top k?", which
is exactly what a reranker can change. `pool_coverage` reports how often the
pool contained anything relevant, so the ceiling is visible.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional


def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def _norm(v) -> str:
    return str(v).strip().lower()


def matches(payload: dict, matcher: dict) -> bool:
    """True if `payload` satisfies every condition in `matcher`."""
    for key, expected in matcher.items():
        if key == "doc_type":
            if _norm(payload.get("doc_type", "")) != _norm(expected):
                return False
        elif key == "act":
            want = _norm(expected)
            act = _norm(payload.get("act", ""))
            act_name = _norm(payload.get("act_name", ""))
            if not (act == want or (want and want in act_name)):
                return False
        elif key == "section_number":
            got = _norm(payload.get("section_number", ""))
            if got not in {_norm(x) for x in _as_list(expected)}:
                return False
        elif key == "title_contains":
            hay = _norm(payload.get("title", "")) + " " + _norm(payload.get("section_title", ""))
            if not any(_norm(x) in hay for x in _as_list(expected)):
                return False
        elif key == "speaker_contains":
            hay = _norm(payload.get("speaker", ""))
            if not any(_norm(x) in hay for x in _as_list(expected)):
                return False
        elif key == "text_contains_any":
            hay = _norm(payload.get("text", ""))
            if not any(_norm(x) in hay for x in _as_list(expected)):
                return False
        elif key == "text_contains_all":
            hay = _norm(payload.get("text", ""))
            if not all(_norm(x) in hay for x in _as_list(expected)):
                return False
        else:
            raise ValueError(f"Unknown matcher key: {key}")
    return True


def is_relevant(payload: dict, matchers: Iterable[dict]) -> bool:
    return any(matches(payload, m) for m in matchers)


def relevance_flags(candidates: list, matchers: list) -> list:
    """Binary relevance for each ranked candidate (objects with .payload)."""
    return [is_relevant(c.payload, matchers) for c in candidates]


# ─────────────────────────────────────────────────────────
# Rank metrics (operate on a ranked list of booleans)
# ─────────────────────────────────────────────────────────

def hit_at_k(rels: list, k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def precision_at_k(rels: list, k: int) -> float:
    top = rels[:k]
    return (sum(top) / len(top)) if top else 0.0


def recall_at_k(rels: list, k: int, pool_relevant: int) -> Optional[float]:
    """Pooled recall. None when the pool holds no relevant chunk (undefined)."""
    if pool_relevant <= 0:
        return None
    return min(sum(rels[:k]), pool_relevant) / pool_relevant


def reciprocal_rank(rels: list, k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: list, k: int, pool_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels[:k]) if r)
    ideal_n = min(k, max(pool_relevant, sum(rels[:k])))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def query_metrics(rels: list, ks: list, pool_relevant: int) -> dict:
    """All per-query metrics at each cutoff in `ks`."""
    out: dict = {"pool_relevant": pool_relevant,
                 "pool_hit": 1.0 if pool_relevant > 0 else 0.0}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(rels, k)
        out[f"precision@{k}"] = precision_at_k(rels, k)
        out[f"recall@{k}"] = recall_at_k(rels, k, pool_relevant)
        out[f"mrr@{k}"] = reciprocal_rank(rels, k)
        out[f"ndcg@{k}"] = ndcg_at_k(rels, k, pool_relevant)
    return out


def mean_ignoring_none(values: list) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def aggregate(per_query: list) -> dict:
    """Mean of every numeric metric across queries (None values are skipped)."""
    if not per_query:
        return {}
    keys = [k for k in per_query[0] if k not in ("pool_relevant",)]
    return {k: mean_ignoring_none([q.get(k) for q in per_query]) for k in keys}


def percentile(values: list, p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (p / 100.0)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
