"""Second-opinion check on whether retrieved passages actually answer the
question, used for the grey zone where embedding similarity cannot decide.

Why this exists (2026-10-13, scripts/probe_unanswerable.py): with exact
cosine scores, clearly off-topic questions reach dense 0.82-0.855 ("flood
insurance claim", "learn to swim", "snake bite") -- inside the range real
answerable questions occupy (~0.85-0.895). multilingual-e5-small's scores are
compressed, so a similarity floor alone either answers off-topic questions
or refuses valid ones.

This is an LLM judgement, not a calibrated probability, and CLAUDE.md SS8.1
warns an LLM verifier is not automatically reliable: it is only enabled for
the grey zone and its accuracy is measured by scripts/eval_evidence_check.py.
"""

from disaster_assistant.generation import ollama_client
from disaster_assistant.retrieval.hybrid import RetrievedChunk

_PROMPT = """You are checking whether some reference passages can answer a question. \
Do NOT answer the question.

Question: {question}

Passages:
{passages}

Do the passages contain information that directly answers this specific question? A passage about a related topic that does not address what was asked counts as NO. Reply with exactly one word: YES or NO."""


_LENIENT_PROMPT = """You are checking whether some reference passages can help with a question. Do NOT answer the question.

Question: {question}

Passages:
{passages}

Do the passages contain safety guidance that helps with this specific situation, even if it is only part of what a full answer would need? A passage about a different topic, or one that merely shares a few words with the question, counts as NO. These passages only cover physical safety actions: questions \nabout money, insurance, compensation, legal or administrative procedures, or medical treatment count as NO. \nReply with exactly one word: YES or NO."""

# query substring -> hazard categories whose passages are on-topic for it
_HAZARD_CATEGORIES = {
    "flood": ("flood", "urban_flood"), "cyclone": ("cyclone",), "earthquake": ("earthquake",),
    "quake": ("earthquake",), "tsunami": ("tsunami",), "landslide": ("landslide",),
    "lightning": ("lightning",), "fire": ("fire", "forest_fire"), "heat": ("heat_wave",),
    "cold": ("cold_wave",), "storm": ("thunderstorm_dust_storm_squall",), "drought": ("drought",),
}


def _hazard_matches_top_passages(question: str, results: list[RetrievedChunk]) -> bool:
    q = question.lower()
    top_categories = {r.disaster_category for r in results[:3]}
    return any(key in q and top_categories & set(cats) for key, cats in _HAZARD_CATEGORIES.items())


def evidence_supported(question: str, results: list[RetrievedChunk]) -> bool:
    """Strict check first. If it says NO but the question names a hazard AND the
    top passages come from that hazard's category, ask a more lenient version
    ("helps with this situation even if partial"). Measured 2026-10-13: the
    lenient prompt alone fixes the flagship flood question but also lets
    off-topic questions through (snake bite, driving test), so it is only
    allowed when hazard + passage category agree."""
    if evidence_supports_question(question, results):
        return True
    if not _hazard_matches_top_passages(question, results):
        return False
    passages = "\n\n".join(f"[{i}] {r.text}" for i, r in enumerate(results[:5], start=1))
    reply = ollama_client.generate(_LENIENT_PROMPT.format(question=question, passages=passages), max_tokens=6)
    return reply.strip().upper().startswith("YES")


def evidence_supports_question(question: str, results: list[RetrievedChunk], max_passages: int = 5) -> bool:
    passages = "\n\n".join(f"[{i}] {r.text}" for i, r in enumerate(results[:max_passages], start=1))
    reply = ollama_client.generate(_PROMPT.format(question=question, passages=passages), max_tokens=6)
    return reply.strip().upper().startswith("YES")
