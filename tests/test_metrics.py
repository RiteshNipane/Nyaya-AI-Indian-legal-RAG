import math

import pytest

from evaluation.metrics import (
    aggregate, hit_at_k, matches, ndcg_at_k, percentile, precision_at_k,
    query_metrics, recall_at_k, reciprocal_rank, relevance_flags,
)
from tests.fakes import act, judgment


def test_matcher_and_semantics():
    p = {"doc_type": "act", "act": "bns", "section_number": 103, "text": "Whoever commits MURDER"}
    assert matches(p, {"doc_type": "act", "act": "bns", "section_number": [101, 103]})
    assert not matches(p, {"doc_type": "act", "act": "ipc"})
    assert not matches(p, {"doc_type": "act", "section_number": [101]})
    assert matches(p, {"text_contains_any": ["murder"]})          # case-insensitive
    assert not matches(p, {"text_contains_all": ["murder", "theft"]})


def test_section_number_string_vs_int_and_suffix():
    assert matches({"section_number": "304B"}, {"section_number": ["304b"]})
    assert matches({"section_number": "63"}, {"section_number": [63]})


def test_act_matches_act_name_substring():
    p = {"act": "rti", "act_name": "Right to Information Act, 2005"}
    assert matches(p, {"act": "right to information"})


def test_unknown_matcher_key_raises():
    with pytest.raises(ValueError):
        matches({}, {"nope": 1})


def test_rank_metrics_known_values():
    rels = [False, True, False, True]
    assert hit_at_k(rels, 1) == 0.0 and hit_at_k(rels, 2) == 1.0
    assert precision_at_k(rels, 4) == 0.5
    assert reciprocal_rank(rels, 4) == 0.5
    assert reciprocal_rank([False, False], 2) == 0.0
    assert recall_at_k(rels, 2, pool_relevant=2) == 0.5
    assert recall_at_k(rels, 4, pool_relevant=0) is None


def test_ndcg_perfect_and_worse_ordering():
    assert ndcg_at_k([True, True, False], 3, pool_relevant=2) == pytest.approx(1.0)
    good = ndcg_at_k([True, False, False], 3, pool_relevant=1)
    bad = ndcg_at_k([False, False, True], 3, pool_relevant=1)
    assert good == pytest.approx(1.0)
    assert bad == pytest.approx(1 / math.log2(4))
    assert good > bad


def test_ndcg_zero_when_nothing_relevant():
    assert ndcg_at_k([False, False], 2, pool_relevant=0) == 0.0


def test_query_metrics_keys_and_pool_flags():
    m = query_metrics([True, False], [1, 2], pool_relevant=2)
    assert m["pool_hit"] == 1.0
    assert {"hit@1", "precision@2", "recall@2", "mrr@1", "ndcg@2"} <= set(m)
    assert query_metrics([False], [1], pool_relevant=0)["pool_hit"] == 0.0


def test_aggregate_skips_none():
    rows = [{"recall@1": 1.0, "hit@1": 1.0, "pool_relevant": 3},
            {"recall@1": None, "hit@1": 0.0, "pool_relevant": 0}]
    agg = aggregate(rows)
    assert agg["recall@1"] == 1.0 and agg["hit@1"] == 0.5
    assert "pool_relevant" not in agg


def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == pytest.approx(25)
    assert percentile([5], 95) == 5
    assert percentile([], 50) == 0.0


def test_relevance_flags_on_candidates():
    cands = [act("a", 103, 0.9), judgment("j", "X v Y", 0.8, "basic structure")]
    flags = relevance_flags(cands, [{"doc_type": "act", "section_number": [103]}])
    assert flags == [True, False]
