from retrieval.multilingual import LLMTranslator
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_classifier import QueryClassifier
from retrieval.reranker import BM25Reranker, FallbackReranker
from tests.fakes import FakeEmbedder, FakeStore, act, cad, judgment
from tests.test_multilingual import FakeClient


def corpus():
    return [
        judgment("j1", "Some v. State", 0.95, "bail granted on grounds of parity"),
        judgment("j2", "Other v. Union", 0.60, "basic structure doctrine"),
        act("a103", 103, 0.70, "Whoever commits murder shall be punished with death",
            title="Punishment for murder"),
        act("a303", 303, 0.65, "Whoever commits theft shall be punished", title="Theft"),
        cad("c1", "Dr. Ambedkar", 0.55, "directive principles debate"),
    ]


def make(reranker=None, translator=None, **kw):
    store = FakeStore(corpus())
    emb = FakeEmbedder()
    p = RetrievalPipeline(emb, store, reranker=reranker, classifier=QueryClassifier(),
                          translator=translator, candidate_k=10, **kw)
    return p, store, emb


def test_baseline_is_dense_order_with_no_routing():
    p, store, _ = make()
    res = p.retrieve("anything at all", top_k=3, rerank=False, route=False)
    assert [c.id for c in res.candidates] == ["j1", "a103", "a303"]
    assert res.routed_doc_types is None and res.reranker is None
    assert store.calls[0]["limit"] == 3  # no wide pool without a reranker


def test_routing_restricts_doc_types():
    p, store, _ = make(min_results=1)
    res = p.retrieve("What is the punishment for murder under BNS section 103?",
                     top_k=2, rerank=False, route=True)
    assert res.routed_doc_types == ["act"]
    assert {c.payload["doc_type"] for c in res.candidates} == {"act"}
    assert res.classification.query_type == "statute_lookup"
    assert not res.widened


def test_explicit_doc_type_overrides_classifier_and_never_widens():
    p, store, _ = make(min_results=5)
    res = p.retrieve("punishment for murder under BNS section 103", top_k=5,
                     doc_type="cad", rerank=False)
    assert res.routed_doc_types == ["cad"] and res.classification is None
    assert not res.widened and len(store.calls) == 1


def test_widens_when_routed_search_is_too_thin():
    p, store, _ = make(min_results=3)
    # only one cad chunk exists -> routed pool (1) < min_results (3)
    res = p.retrieve("What did Ambedkar say in the Constituent Assembly?",
                     top_k=3, rerank=False, route=True)
    assert res.widened and len(store.calls) == 2
    assert store.calls[1]["doc_types"] is None
    assert len({c.id for c in res.pool}) == len(res.pool)  # deduped
    assert len(res.candidates) == 3


def test_reranker_reorders_and_uses_wide_pool():
    p, store, _ = make(reranker=BM25Reranker())
    res = p.retrieve("punishment for murder", top_k=2, rerank=True, route=False)
    assert res.candidates[0].id == "a103"          # dense had j1 first
    assert res.reranker == "bm25"
    assert store.calls[0]["limit"] == 10           # candidate_k pool
    assert len(res.pool) == 5


def test_rerank_flag_off_skips_reranker_even_if_configured():
    p, _, _ = make(reranker=BM25Reranker())
    res = p.retrieve("punishment for murder", top_k=2, rerank=False, route=False)
    assert res.candidates[0].id == "j1" and res.reranker is None


def test_reranker_failure_degrades_to_dense_order():
    class Boom:
        name = "boom"

        def rerank(self, *a, **k):
            raise RuntimeError("api down")

    p, _, _ = make(reranker=FallbackReranker([Boom()]))
    res = p.retrieve("punishment for murder", top_k=2, rerank=True, route=False)
    assert [c.id for c in res.candidates] == ["j1", "a103"]
    assert res.reranker == "none"


def test_non_english_query_is_translated_before_embedding_and_routing():
    client = FakeClient(reply="What is the punishment for murder?")
    p, _, emb = make(translator=LLMTranslator(client, "m"), min_results=1)
    res = p.retrieve("हत्या की सजा क्या है?", top_k=2, rerank=False, route=True)
    assert emb.queries == ["What is the punishment for murder?"]
    assert res.language.code == "hi"
    assert res.retrieval_query == "What is the punishment for murder?"
    assert res.routed_doc_types == ["act"]


def test_timings_present():
    p, _, _ = make(reranker=BM25Reranker())
    res = p.retrieve("murder", top_k=2)
    assert {"translate", "classify", "embed", "search", "rerank", "total"} <= set(res.timings_ms)
