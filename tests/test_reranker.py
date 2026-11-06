import pytest

from retrieval import reranker as rr
from retrieval.reranker import (
    BM25Reranker, CohereReranker, FallbackReranker, get_reranker,
)
from tests.fakes import act, judgment


def pool():
    return [
        judgment("j1", "Unrelated v. State", 0.90, "land acquisition compensation award"),
        act("a103", 103, 0.80, "Whoever commits murder shall be punished with death or "
                              "imprisonment for life", title="Punishment for murder"),
        act("a303", 303, 0.70, "Whoever commits theft shall be punished", title="Theft"),
    ]


def test_bm25_reorders_by_query_terms():
    out = BM25Reranker().rerank("punishment for murder", pool(), top_k=3)
    assert out[0].id == "a103"
    assert all(0.0 <= c.rerank_score <= 1.0 for c in out)
    assert out[0].rerank_score == pytest.approx(1.0)


def test_bm25_respects_top_k_and_does_not_mutate_input():
    original = pool()
    out = BM25Reranker().rerank("murder", original, top_k=1)
    assert len(out) == 1
    assert all(c.rerank_score is None for c in original)


def test_bm25_empty_and_no_overlap():
    assert BM25Reranker().rerank("x", [], 5) == []
    out = BM25Reranker().rerank("zzzz", pool(), 3)
    assert [c.id for c in out] == ["j1", "a103", "a303"]  # stable dense order


class FakeResp:
    def __init__(self, status, data=None, text=""):
        self.status_code, self._data, self.text = status, data, text

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


def test_cohere_orders_by_relevance_and_sends_expected_request():
    session = FakeSession([FakeResp(200, {"results": [
        {"index": 1, "relevance_score": 0.97},
        {"index": 0, "relevance_score": 0.12},
    ]})])
    r = CohereReranker(api_key="k", session=session)
    out = r.rerank("punishment for murder", pool(), top_k=2)

    assert [c.id for c in out] == ["a103", "j1"]
    assert out[0].rerank_score == pytest.approx(0.97)
    call = session.calls[0]
    assert call["url"].endswith("/v2/rerank")
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["json"]["top_n"] == 2 and len(call["json"]["documents"]) == 3
    assert "BNS, Section 103" in call["json"]["documents"][1]


def test_cohere_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    session = FakeSession([FakeResp(429), FakeResp(200, {"results": [
        {"index": 0, "relevance_score": 0.5}]})])
    out = CohereReranker(api_key="k", session=session).rerank("q", pool(), 1)
    assert len(session.calls) == 2 and out[0].id == "j1"


def test_cohere_client_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    session = FakeSession([FakeResp(401, text="bad key")])
    with pytest.raises(RuntimeError, match="401"):
        CohereReranker(api_key="k", session=session).rerank("q", pool(), 1)
    assert len(session.calls) == 1


def test_cohere_requires_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        CohereReranker()


def test_fallback_uses_next_reranker_on_failure():
    class Boom:
        name = "boom"

        def rerank(self, *a, **k):
            raise RuntimeError("down")

    chain = FallbackReranker([Boom(), BM25Reranker()])
    out = chain.rerank("murder", pool(), 2)
    assert out[0].id == "a103" and chain.last_used == "bm25"
    assert chain.name == "boom+bm25"


def test_fallback_keeps_dense_order_when_everything_fails():
    class Boom:
        name = "boom"

        def rerank(self, *a, **k):
            raise RuntimeError("down")

    chain = FallbackReranker([Boom()])
    out = chain.rerank("q", pool(), 2)
    assert [c.id for c in out] == ["j1", "a103"] and chain.last_used == "none"


def test_get_reranker_selection(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "k")
    monkeypatch.setattr(rr.importlib.util, "find_spec", lambda name: None)
    assert get_reranker("none") is None
    assert get_reranker("bm25").name == "bm25"
    assert get_reranker("auto").name == "cohere+bm25"
    monkeypatch.delenv("COHERE_API_KEY")
    assert get_reranker("auto").name == "bm25"
    monkeypatch.setattr(rr.importlib.util, "find_spec", lambda name: object())
    assert get_reranker("auto").name == "cross-encoder+bm25"
    with pytest.raises(ValueError):
        get_reranker("nonsense")
