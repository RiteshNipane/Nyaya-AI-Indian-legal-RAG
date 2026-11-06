import pytest

from evaluation.generation_eval import (
    citation_metrics, evaluate_answer, extract_citations, keyword_coverage,
    parse_judge_json,
)
from retrieval.generation import build_prompt
from retrieval.multilingual import language_from_code
from tests.fakes import act


def test_extract_citations_handles_multiple_formats():
    ans = "Murder is punished [Source 1]. Also see [Sources 3, 2] and [Source 2; Source 4]."
    assert extract_citations(ans) == {1, 2, 3, 4}
    assert extract_citations("no citations here") == set()


def test_citation_metrics_flags_invalid_numbers():
    m = citation_metrics("A [Source 1] B [Source 9]", n_sources=3)
    assert m["has_citation"] == 1.0
    assert m["citation_validity"] == pytest.approx(0.5)
    assert m["invalid_citations"] == [9]
    assert m["source_coverage"] == pytest.approx(1 / 3)


def test_citation_metrics_without_citations():
    m = citation_metrics("plain answer", n_sources=3)
    assert m["has_citation"] == 0.0 and m["citation_validity"] == 0.0


def test_keyword_coverage():
    assert keyword_coverage("Section 103 punishes murder", ["murder", "103", "dowry"]) == pytest.approx(2 / 3)
    assert keyword_coverage("x", None) is None


def test_parse_judge_json():
    assert parse_judge_json('{"score": 4, "reason": "ok"}')["score"] == 4
    assert parse_judge_json('Sure!\n```json\n{"score": 2, "reason": "weak"}\n```')["score"] == 2
    assert parse_judge_json('{"score": 9}') is None
    assert parse_judge_json("no json") is None
    assert parse_judge_json("") is None


def test_evaluate_answer_without_judge():
    out = evaluate_answer("q", "Murder [Source 1]", ["text"], ["murder"])
    assert out["keyword_coverage"] == 1.0 and "faithfulness" not in out


def test_build_prompt_includes_sources_and_language_note():
    prompt = build_prompt("q?", [act("a", 103, 0.9, "murder text", title="Murder")],
                          language_from_code("hi"))
    assert "[Source 1: BNS, Section 103 — Murder]" in prompt
    assert "Hindi" in prompt
    assert "Hindi" not in build_prompt("q?", [act("a", 103, 0.9, "t")], language_from_code("en"))
