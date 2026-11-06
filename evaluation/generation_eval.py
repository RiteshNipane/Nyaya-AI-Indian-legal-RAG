"""
evaluation/generation_eval.py

Answer-quality checks.

Deterministic (free, always on):
  * citation_metrics   – does the answer cite [Source N], are the N valid,
                         how much of the context is actually cited
  * keyword_coverage   – fraction of expected keywords/sections present

LLM-as-judge (optional, needs an OpenAI-compatible client):
  * faithfulness       – is every claim supported by the provided sources?
  * answer_relevance   – does the answer address the question?
  Scores are 1-5 from the judge and normalised to 0-1.
"""

from __future__ import annotations

import json
import re
from typing import Optional

_CITATION_BLOCK = re.compile(r"\[\s*Sources?\s+([^\]]+)\]", re.IGNORECASE)


def extract_citations(answer: str) -> set:
    """Set of source numbers cited, e.g. {1, 3} for '[Source 1]' and '[Sources 3, 1]'."""
    cited: set = set()
    for m in _CITATION_BLOCK.finditer(answer or ""):
        cited.update(int(n) for n in re.findall(r"\d+", m.group(1)))
    return cited


def citation_metrics(answer: str, n_sources: int) -> dict:
    cited = extract_citations(answer)
    valid = {n for n in cited if 1 <= n <= n_sources}
    return {
        "has_citation": 1.0 if cited else 0.0,
        "citation_validity": (len(valid) / len(cited)) if cited else 0.0,
        "source_coverage": (len(valid) / n_sources) if n_sources else 0.0,
        "invalid_citations": sorted(cited - valid),
    }


def keyword_coverage(answer: str, keywords: Optional[list]) -> Optional[float]:
    if not keywords:
        return None
    text = (answer or "").lower()
    return sum(1 for k in keywords if str(k).lower() in text) / len(keywords)


JUDGE_SYSTEM = (
    "You are a strict evaluator of answers produced by a legal RAG system. "
    "Reply with ONLY a JSON object: {\"score\": <integer 1-5>, \"reason\": \"<one sentence>\"}."
)

FAITHFULNESS_PROMPT = """Rate FAITHFULNESS from 1 to 5.
5 = every factual claim is supported by the SOURCES; 3 = some claims unsupported;
1 = mostly unsupported or contradicts the sources. Ignore style.

QUESTION: {question}

SOURCES:
{sources}

ANSWER:
{answer}"""

RELEVANCE_PROMPT = """Rate ANSWER RELEVANCE from 1 to 5.
5 = directly and completely addresses the question; 3 = partially; 1 = off-topic or a refusal
that was not warranted.

QUESTION: {question}

ANSWER:
{answer}"""


def parse_judge_json(text: str) -> Optional[dict]:
    """Extract {"score": n} from a judge reply, tolerating code fences and prose."""
    if not text:
        return None
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        score = int(data["score"])
    except (ValueError, KeyError, TypeError):
        return None
    if not 1 <= score <= 5:
        return None
    return {"score": score, "reason": str(data.get("reason", ""))}


class LLMJudge:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def _ask(self, prompt: str) -> Optional[dict]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=200, temperature=0,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": prompt}],
            )
            return parse_judge_json(resp.choices[0].message.content or "")
        except Exception:
            return None

    def faithfulness(self, question: str, answer: str, source_texts: list) -> Optional[float]:
        sources = "\n\n".join(f"[Source {i}] {t[:1200]}" for i, t in enumerate(source_texts, 1))
        r = self._ask(FAITHFULNESS_PROMPT.format(question=question, sources=sources, answer=answer))
        return (r["score"] - 1) / 4 if r else None

    def answer_relevance(self, question: str, answer: str) -> Optional[float]:
        r = self._ask(RELEVANCE_PROMPT.format(question=question, answer=answer))
        return (r["score"] - 1) / 4 if r else None


def evaluate_answer(question: str, answer: str, source_texts: list,
                    expected_keywords: Optional[list] = None,
                    judge: Optional[LLMJudge] = None) -> dict:
    out = citation_metrics(answer, len(source_texts))
    out["keyword_coverage"] = keyword_coverage(answer, expected_keywords)
    if judge is not None:
        out["faithfulness"] = judge.faithfulness(question, answer, source_texts)
        out["answer_relevance"] = judge.answer_relevance(question, answer)
    return out
