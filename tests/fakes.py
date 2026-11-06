"""Tiny in-memory stand-ins for the embedder / vector store used in tests."""
from retrieval.types import Candidate


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def encode_query(self, text):
        self.queries.append(text)
        return [0.0]


class FakeStore:
    """Returns pre-scored candidates, honouring the doc_types filter."""

    def __init__(self, items):
        self.items = items
        self.calls = []

    def search(self, vector, limit, doc_types=None):
        self.calls.append({"limit": limit, "doc_types": doc_types})
        rows = [c for c in self.items
                if not doc_types or c.payload.get("doc_type") in doc_types]
        rows = sorted(rows, key=lambda c: c.score, reverse=True)
        return rows[:limit]


def act(id, section, score, text="", act="bns", title=""):
    return Candidate(id=id, score=score, payload={
        "doc_type": "act", "act": act, "act_name": act.upper(),
        "section_number": section, "section_title": title, "text": text})


def judgment(id, title, score, text=""):
    return Candidate(id=id, score=score, payload={
        "doc_type": "judgment", "title": title, "year": 2023,
        "court": "Supreme Court of India", "text": text})


def cad(id, speaker, score, text=""):
    return Candidate(id=id, score=score, payload={
        "doc_type": "cad", "speaker": speaker, "volume": 7, "date": "1948", "text": text})
