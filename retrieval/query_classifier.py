"""
retrieval/query_classifier.py

Rule-based query classifier that routes a question to the right slice of the
corpus (acts / judgments / Constituent Assembly Debates).

Design notes
------------
* Dependency-free and deterministic: no model call, ~microseconds per query.
* Each doc type accumulates weighted evidence from regex "signal groups".
  A group contributes `weight * min(distinct_matches, cap)`.
* Routing is conservative. We only restrict the search when the evidence is
  strong (top score >= MIN_ROUTE_SCORE). Weak or mixed signals return
  `doc_types=None`, meaning "search everything". The retrieval pipeline also
  widens to the full corpus when a routed search returns too few chunks.
* The Constitution is indexed as its own doc_type ("constitution"). Constitutional
  vocabulary ("Article 21", "fundamental rights", ...) is deliberately too weak to
  route on its own, so a plain constitutional question searches the whole corpus
  (provisions + judgments + CAD). When a stronger signal routes to judgments or
  CAD, the Constitution rides along as a companion type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

DOC_TYPES = ("act", "constitution", "judgment", "cad")
# Types that can drive routing. "constitution" is only ever a companion.
PRIMARY_TYPES = ("act", "judgment", "cad")

# Route only when the strongest type has at least this much evidence.
MIN_ROUTE_SCORE = 2.0
# A secondary type is included when it reaches this share of the top score
# (and at least MIN_SECONDARY_SCORE in absolute terms).
SECONDARY_RATIO = 0.4
MIN_SECONDARY_SCORE = 1.5


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# (compiled regex, weight, cap on distinct matches counted)
SIGNALS: dict[str, list[tuple[re.Pattern, float, int]]] = {
    "act": [
        # explicit section references: "section 103", "s. 302", "sec 41A"
        (_rx(r"\b(?:section|sec\.?|s\.)\s*\d+[a-z]?\b"), 2.0, 1),
        # "which section", "what section", "section of/under"
        (_rx(r"\b(?:which|what)\s+section\b|\bsection\s+(?:of|under)\b"), 2.0, 1),
        # named statutes / codes
        (_rx(
            r"\b(?:bns|bnss|bsa|ipc|crpc|cr\.?\s?p\.?\s?c\.?|cpc|"
            r"bharatiya\s+nyaya|bharatiya\s+nagarik|bharatiya\s+sakshya|"
            r"indian\s+penal\s+code|code\s+of\s+criminal\s+procedure|"
            r"evidence\s+act|it\s+act|information\s+technology\s+act|"
            r"rti|right\s+to\s+information|pocso|"
            r"protection\s+of\s+children\s+from\s+sexual)\b"), 2.0, 1),
        # offence / procedure vocabulary (each distinct term adds 1, up to 3)
        (_rx(
            r"\b(?:punish\w*|penalt\w*|imprisonment|offen[cs]e\w*|cognizable|"
            r"non-?bailable|bailable|bail|anticipatory|arrest\w*|warrant|fir|"
            r"chargesheet|cheating|murder|homicide|theft|rape|robbery|dacoity|"
            r"kidnapp\w*|forgery|defamation|dowry|assault|extortion|bribery|"
            r"sedition|terroris\w*|admissib\w*)\b"), 1.0, 3),
    ],
    "constitution": [
        # weight stays below MIN_ROUTE_SCORE: never routes alone, only rides along
        (_rx(
            r"\b(?:article\s+\d+[a-z]?|fundamental\s+rights?|"
            r"directive\s+principles?|preamble|fundamental\s+duties|"
            r"personal\s+liberty|freedom\s+of\s+speech|"
            r"right\s+to\s+(?:equality|life|freedom)|constitutional|"
            r"(?:the\s+)?constitution(?:\s+of\s+india)?)\b"), 1.5, 1),
    ],
    "judgment": [
        # "X v. Y", "X vs Y", "X versus Y"
        (_rx(r"\b\w+\s+(?:v\.|vs\.?|versus)\s+\w+"), 3.0, 1),
        # courts / case-law vocabulary
        (_rx(r"\b(?:supreme\s+court|high\s+court|apex\s+court)\b"), 2.0, 1),
        (_rx(
            r"\b(?:case\s+law|judg(?:e)?ments?|rulings?|precedents?|landmark|"
            r"verdicts?|writ\s+petition|ratio|bench|held|holding)\b"), 1.5, 2),
        # strong case-law verbs
        (_rx(r"\b(?:struck\s+down|upheld|overruled|read\s+down|quashed?|quashing)\b"), 2.0, 1),
        # well-known doctrines / cases
        (_rx(
            r"\b(?:basic\s+structure|rarest\s+of\s+rare|due\s+process|"
            r"proportionality|locus\s+standi|res\s+judicata|"
            r"public\s+interest\s+litigation|judicial\s+review|"
            r"separation\s+of\s+powers|colou?rable\s+legislation|"
            r"essential\s+religious\s+practices?|right\s+to\s+privacy|"
            r"kesavananda|maneka\s+gandhi|puttaswamy|navtej|"
            r"shreya\s+singhal|vishaka)\b"), 2.0, 1),
        (_rx(r"\bdoctrine\b"), 1.5, 1),
        (_rx(r"\b(?:article\s+\d+[a-z]?)\b"), 0.5, 1),
    ],
    "cad": [
        (_rx(
            r"\b(?:constituent\s+assembly|framers?|drafting\s+committee|"
            r"drafters?|founding\s+fathers|original\s+intent|"
            r"framer(?:s'?)?\s+intent|debates?|debated)\b"), 3.0, 1),
        (_rx(
            r"\b(?:ambedkar|nehru|sardar\s+patel|patel|rajendra\s+prasad|"
            r"munshi|alladi|krishnamachari|ayyar|"
            r"gopalaswami|ayyangar|b\.?\s?n\.?\s?rau)\b"), 3.0, 1),
        (_rx(r"\bwhy\s+did\s+(?:the\s+)?(?:framers?|assembly|drafters?)\b"), 1.5, 1),
    ],
}

QUERY_TYPE_BY_DOC_TYPE = {
    "act": "statute_lookup",
    "constitution": "constitutional",
    "judgment": "case_law",
    "cad": "framer_intent",
}


@dataclass
class QueryClassification:
    query_type: str                      # statute_lookup | case_law | framer_intent | multi | general
    scores: dict = field(default_factory=dict)
    doc_types: Optional[list] = None     # None => search the whole corpus
    confidence: float = 0.0              # top score / total score, 0..1
    signals: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "doc_types": self.doc_types,
            "confidence": round(self.confidence, 3),
            "scores": {k: round(v, 2) for k, v in self.scores.items()},
            "signals": self.signals,
        }


class QueryClassifier:
    """Classify a (preferably English) query and decide which doc types to search."""

    def classify(self, query: str) -> QueryClassification:
        text = (query or "").strip()
        scores = {t: 0.0 for t in DOC_TYPES}
        signals: list[str] = []

        if not text:
            return QueryClassification(query_type="general", scores=scores)

        for doc_type, groups in SIGNALS.items():
            for regex, weight, cap in groups:
                matches = {m.group(0).lower() for m in regex.finditer(text)}
                if not matches:
                    continue
                counted = min(len(matches), cap)
                scores[doc_type] += weight * counted
                signals.append(f"{doc_type}:{sorted(matches)[0]}")

        total = sum(scores.values())
        top_type = max(scores, key=scores.get)
        top_score = scores[top_type]
        confidence = (top_score / total) if total > 0 else 0.0

        if top_score < MIN_ROUTE_SCORE:
            return QueryClassification(
                query_type="general", scores=scores, doc_types=None,
                confidence=confidence, signals=signals,
            )

        threshold = max(MIN_SECONDARY_SCORE, SECONDARY_RATIO * top_score)
        selected = [t for t in DOC_TYPES if scores[t] >= threshold]

        if set(PRIMARY_TYPES) <= set(selected):
            return QueryClassification(
                query_type="multi", scores=scores, doc_types=None,
                confidence=confidence, signals=signals,
            )

        query_type = (
            QUERY_TYPE_BY_DOC_TYPE[selected[0]] if len(selected) == 1 else "multi"
        )
        return QueryClassification(
            query_type=query_type, scores=scores, doc_types=selected,
            confidence=confidence, signals=signals,
        )
