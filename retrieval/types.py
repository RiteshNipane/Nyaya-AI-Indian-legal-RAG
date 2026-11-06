"""
retrieval/types.py

Light, dependency-free data types shared by the API, the retrieval pipeline
and the evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Candidate:
    """One retrieved chunk. `score` is the dense (cosine) similarity."""

    id: str
    payload: dict
    score: float = 0.0
    rerank_score: Optional[float] = None

    @property
    def final_score(self) -> float:
        """Score used for ordering/display: reranker score when present."""
        return self.rerank_score if self.rerank_score is not None else self.score

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


def format_source_label(payload: dict) -> str:
    """Build a readable source label from chunk metadata."""
    doc_type = payload.get("doc_type", "")

    if doc_type == "judgment":
        title = payload.get("title", "Unknown Case")
        year = payload.get("year", "")
        court = payload.get("court", "Supreme Court of India")
        return f"{title} ({year}) — {court}"

    if doc_type == "act":
        act_name = payload.get("act_name") or payload.get("act", "").upper()
        section = payload.get("section_number", "")
        title = payload.get("section_title", "")
        return f"{act_name}, Section {section} — {title}"

    if doc_type == "constitution":
        return "Constitution of India"

    if doc_type == "cad":
        volume = payload.get("volume", "")
        date = payload.get("date", "")
        speaker = payload.get("speaker", "")
        return f"CAD Vol.{volume} ({date}) — {speaker}"

    return payload.get("source", "Legal Source")
