"""
retrieval/generation.py

Prompt construction and LLM answer generation, shared by the API and the
evaluation harness so both exercise exactly the same prompt.
"""

from __future__ import annotations

from typing import Optional

from retrieval.multilingual import LanguageInfo, answer_language_instruction
from retrieval.types import format_source_label

SYSTEM_PROMPT = """You are an expert Indian constitutional lawyer and legal researcher.

You help lawyers, students, and citizens understand Indian law by providing:
1. Relevant constitutional articles and their interpretation
2. Landmark Supreme Court and High Court judgments
3. The legal reasoning behind key principles
4. Framer intent from Constituent Assembly Debates when relevant

STRICT RULES:
- ONLY use information from the provided context documents
- If context is insufficient, say so clearly — never hallucinate cases or articles
- Always cite: case name + year + court for every judgment
- Always cite: Article number for every constitutional provision
- Distinguish binding precedent (Supreme Court) from persuasive authority (High Court)
- Never give direct legal advice — provide legal information only

OUTPUT FORMAT:
1. **Relevant Constitutional Provisions** — Articles that apply
2. **Key Judgments** — Case, year, court, and what was held
3. **Legal Reasoning** — The principle flowing from these sources
4. **Framer Intent** (if available from CAD) — Original intent
5. **Caveats** — Limitations or conflicting views
"""


class GenerationError(RuntimeError):
    """Raised when the LLM call fails."""


def build_prompt(query: str, candidates: list,
                 language: Optional[LanguageInfo] = None) -> str:
    context_parts = []
    for i, c in enumerate(candidates, 1):
        label = format_source_label(c.payload)
        context_parts.append(f"[Source {i}: {label}]\n{c.text}")

    context = "\n\n---\n\n".join(context_parts)
    lang_note = answer_language_instruction(language) if language else ""

    return f"""Answer this legal query using ONLY the sources provided below.

QUERY: {query}

LEGAL SOURCES:
{context}

Provide a structured answer. Cite sources as [Source N].{lang_note}"""


def generate_answer(client, model: str, query: str, candidates: list,
                    language: Optional[LanguageInfo] = None,
                    max_tokens: int = 2000) -> str:
    if client is None:
        raise GenerationError("OPENROUTER_API_KEY is not configured")
    prompt = build_prompt(query, candidates, language)
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise GenerationError(str(e)) from e
