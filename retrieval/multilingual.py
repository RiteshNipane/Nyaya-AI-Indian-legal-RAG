"""
retrieval/multilingual.py

Multilingual support for NyayaAI (Hindi, Marathi, Tamil + other Indian scripts).

Strategy: keep the English-only index and embedding model. A non-English query
is (1) detected by Unicode script, (2) translated to English by an LLM for
retrieval / classification / reranking, and (3) the final answer is generated
in the user's language (statute names, section numbers and case names stay in
English so citations remain verifiable).

If you later move to a multilingual embedding model (e.g. BAAI/bge-m3), set
EMBEDDING_MODEL / VECTOR_SIZE and re-ingest; translation then becomes optional.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("nyaya.multilingual")

# (script name, start, end)
_SCRIPT_RANGES = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali",    0x0980, 0x09FF),
    ("gurmukhi",   0x0A00, 0x0A7F),
    ("gujarati",   0x0A80, 0x0AFF),
    ("odia",       0x0B00, 0x0B7F),
    ("tamil",      0x0B80, 0x0BFF),
    ("telugu",     0x0C00, 0x0C7F),
    ("kannada",    0x0C80, 0x0CFF),
    ("malayalam",  0x0D00, 0x0D7F),
    ("arabic",     0x0600, 0x06FF),
]

_SCRIPT_TO_LANG = {
    "bengali": "bn", "gurmukhi": "pa", "gujarati": "gu", "odia": "or",
    "tamil": "ta", "telugu": "te", "kannada": "kn", "malayalam": "ml",
    "arabic": "ur",
}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "ta": "Tamil",
    "bn": "Bengali", "pa": "Punjabi", "gu": "Gujarati", "or": "Odia",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "ur": "Urdu",
}

# Devanagari is shared by Hindi and Marathi; separate them with common function words.
_MARATHI_MARKERS = {"आहे", "आहेत", "काय", "आणि", "नाही", "मध्ये", "कोणती",
                    "कोणत्या", "होते", "म्हणजे", "किंवा", "तर", "साठी", "ची", "चा", "चे"}
_HINDI_MARKERS = {"है", "हैं", "क्या", "और", "नहीं", "में", "कौन", "कौनसी",
                  "को", "से", "था", "थे", "होता", "होती", "या", "तो", "कि"}
_MARATHI_CHARS = {"ळ"}

_PUNCT = re.compile(r"[\s\?\!\.\,\;\:\"'()\[\]।॥]+")


@dataclass
class LanguageInfo:
    code: str = "en"
    name: str = "English"
    script: str = "latin"
    confidence: float = 1.0

    @property
    def is_english(self) -> bool:
        return self.code == "en"

    def to_dict(self) -> dict:
        return {"code": self.code, "name": self.name,
                "script": self.script, "confidence": round(self.confidence, 2)}


def _script_of(ch: str) -> Optional[str]:
    cp = ord(ch)
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    return None


def detect_language(text: str) -> LanguageInfo:
    """Detect the query language from its dominant script (plus Hindi/Marathi cues)."""
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return LanguageInfo()

    counts: dict = {}
    for ch in letters:
        s = _script_of(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1

    if not counts:
        return LanguageInfo()

    script, n = max(counts.items(), key=lambda kv: kv[1])
    share = n / len(letters)
    if share < 0.3:
        return LanguageInfo(code="en", confidence=1 - share)

    if script == "devanagari":
        tokens = {t for t in _PUNCT.split(text) if t}
        mr = len(tokens & _MARATHI_MARKERS) + (2 if any(c in text for c in _MARATHI_CHARS) else 0)
        hi = len(tokens & _HINDI_MARKERS)
        code = "mr" if mr > hi else "hi"  # ties default to Hindi
        total = mr + hi
        conf = 0.6 if total == 0 else max(mr, hi) / total * 0.5 + 0.5
        return LanguageInfo(code=code, name=LANGUAGE_NAMES[code],
                            script=script, confidence=conf)

    code = _SCRIPT_TO_LANG[script]
    return LanguageInfo(code=code, name=LANGUAGE_NAMES[code],
                        script=script, confidence=share)


def language_from_code(code: Optional[str]) -> Optional[LanguageInfo]:
    """Build LanguageInfo from an explicit ISO code supplied by the client."""
    if not code:
        return None
    code = code.strip().lower()
    if code not in LANGUAGE_NAMES:
        return None
    return LanguageInfo(code=code, name=LANGUAGE_NAMES[code],
                        script="explicit", confidence=1.0)


def answer_language_instruction(lang: LanguageInfo) -> str:
    """Extra prompt text asking the LLM to answer in the user's language."""
    if lang.is_english:
        return ""
    return (
        f"\n\nLANGUAGE: The user asked in {lang.name}. Write the answer in {lang.name}. "
        "Keep statute names, Section/Article numbers and case names in their original "
        "English form so citations stay verifiable, and keep [Source N] markers unchanged."
    )


class LLMTranslator:
    """Translate a query to English with an OpenAI-compatible chat client."""

    SYSTEM = (
        "You translate legal questions into English for a search system. "
        "Preserve statute names, section/article numbers and case names exactly. "
        "Output ONLY the English translation, with no commentary."
    )

    def __init__(self, client, model: str, max_cache: int = 512):
        self.client = client
        self.model = model
        self._cache: dict = {}
        self._max_cache = max_cache

    def to_english(self, text: str, lang: LanguageInfo) -> str:
        key = (lang.code, text)
        if key in self._cache:
            return self._cache[key]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=300,
                temperature=0,
                messages=[
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user",
                     "content": f"Language: {lang.name}\nQuestion: {text}"},
                ],
            )
            out = (resp.choices[0].message.content or "").strip().strip('"')
            if not out:
                return text
        except Exception as e:
            logger.warning("Translation failed (%s); using original query", e)
            return text
        if len(self._cache) >= self._max_cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = out
        return out


def prepare_query(query: str, translator=None,
                  language: Optional[str] = None) -> tuple:
    """Return (retrieval_query_in_english, LanguageInfo).

    `language` is an optional explicit ISO code that overrides detection.
    Without a translator, the original text is used for retrieval.
    """
    info = language_from_code(language) or detect_language(query)
    if info.is_english or translator is None:
        return query, info
    return translator.to_english(query, info), info
