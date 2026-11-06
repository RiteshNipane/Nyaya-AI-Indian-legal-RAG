import pytest

from evaluation.run_eval import DEFAULT_GOLDEN, classifier_report, load_golden
from retrieval.query_classifier import QueryClassifier

clf = QueryClassifier()


@pytest.mark.parametrize("q, expected", [
    ("What is BNS section 103?", ["act"]),
    ("Explain Section 482 of BNSS", ["act"]),
    ("Kesavananda Bharati v. State of Kerala", ["judgment"]),
    ("What did the Supreme Court hold in Maneka Gandhi?", ["judgment"]),
    ("What did Nehru say in the Constituent Assembly?", ["cad"]),
    ("Constituent Assembly debate on Article 21", ["constitution", "cad"]),
])
def test_routes_clear_queries(q, expected):
    assert clf.classify(q).doc_types == expected


@pytest.mark.parametrize("q", [
    "Tell me about fundamental rights",
    "What is Article 21?",
    "right to equality",
    "hello",
    "",
])
def test_weak_or_ambiguous_queries_are_not_routed(q):
    c = clf.classify(q)
    assert c.doc_types is None


def test_mixed_statute_and_case_law_selects_both():
    c = clf.classify("Section 66A IT Act was struck down by the Supreme Court")
    assert set(c.doc_types) == {"act", "judgment"}
    assert c.query_type == "multi"


def test_all_three_types_means_no_filter():
    # balanced evidence: act 4.0, judgment 3.5, cad 3.0 -> all clear the cutoff
    c = clf.classify("BNS section 21, Supreme Court judgment, and Ambedkar")
    assert c.doc_types is None
    assert c.query_type == "multi"


def test_weaker_type_below_cutoff_is_dropped():
    # cad dominates (6.0); act (2.0) is under 40% of the top score
    c = clf.classify("Section 21 supreme court judgment and Ambedkar in the Constituent Assembly debates")
    assert c.doc_types == ["judgment", "cad"]


def test_classification_is_serialisable_and_reports_signals():
    c = clf.classify("punishment for murder under BNS section 103")
    d = c.to_dict()
    assert d["query_type"] == "statute_lookup"
    assert d["signals"] and d["scores"]["act"] > d["scores"]["judgment"]


def test_golden_set_is_well_formed_and_routed_correctly():
    golden = load_golden(DEFAULT_GOLDEN)
    assert len(golden) >= 25
    ids = [g["id"] for g in golden]
    assert len(ids) == len(set(ids))
    rep = classifier_report(golden)
    assert rep["accuracy"] >= 0.9, rep["mismatches"]


def test_constitution_never_routes_alone():
    for q in ["What is Article 21?", "Tell me about the Constitution of India",
              "fundamental rights and directive principles"]:
        c = clf.classify(q)
        assert c.doc_types is None and c.scores["constitution"] >= 1.5


def test_constitution_rides_along_with_case_law():
    c = clf.classify("What did the Supreme Court hold about Article 21?")
    assert c.doc_types == ["constitution", "judgment"]


def test_constitution_dropped_when_other_signal_is_much_stronger():
    c = clf.classify("Why did the framers include the Preamble?")
    assert c.doc_types == ["cad"]
