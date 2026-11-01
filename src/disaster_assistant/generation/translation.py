"""Language auto-detection + translation of a non-English question to
English before retrieval.

Why translate-then-retrieve exists (2026-10-10, see
docs/implementation_status.md Finding #14): direct cross-lingual dense
retrieval was unreliable for specific concepts in Hindi/Telugu/Kannada/
Malayalam -- an earthquake question retrieved the same wrong, generic
passage in all four languages, while flood questions worked fine in the
same languages. Translating to English first and retrieving in English
(where BM25 also finally works, instead of being structurally impossible
across scripts) is the fix CLAUDE.md itself proposed investigating
(SS8.4: "Compare direct multilingual retrieval against translation-based
retrieval").

Why language detection is Unicode-range-based, not an LLM call (found
2026-10-10): the obvious approach -- asking the LLM to detect the
language and translate in one call -- actually misidentified Tamil text
as Telugu (confirmed directly: the model's raw output was
"te|What should I do if the flood comes into the house?" for a Tamil
sentence, correct translation, wrong language code). But each of these
6 languages' everyday text occupies its own disjoint Unicode block --
Devanagari (Hindi), Tamil, Telugu, Kannada, Malayalam, and Latin (English)
don't overlap at all -- so counting which block a question's characters
fall into is both deterministic (no model-confusion risk) and instant
(no extra LLM round-trip), strictly better than asking the model for
something a lookup table already answers exactly.
"""

import re

from disaster_assistant.generation import ollama_client
from disaster_assistant.generation.prompts import LANGUAGE_NAMES

_TRANSLATE_PROMPT = """Translate the following {source_language} text into English. Translate literally. If the text names a natural hazard, use its standard English name (flood, cyclone, earthquake, tsunami, landslide, lightning, fire, heat wave) -- never transliterate it as if it were a name (e.g. Hindi "लू", Telugu "వడగాలులు", Kannada "ಬಿಸಿಗಾಳಿ" and Tamil "வெப்ப அலை" all mean heat wave -- not a person, not dust storms, not hot wind). The text may be about ANY topic and may come from imperfect speech recognition: if it is unclear, garbled, or not about a disaster, translate exactly what is there and do NOT guess, complete, or reinterpret it into a disaster question. Output ONLY the English translation, nothing else -- no quotes, no explanation, no commentary.

{source_language} text: {text}

English translation:"""

# (language_code, regex matching one character of that script). Order
# doesn't matter -- the ranges are disjoint by construction (Unicode
# assigns each of these scripts its own block).
_SCRIPT_PATTERNS = [
    ("hi", re.compile(r"[ऀ-ॿ]")),  # Devanagari
    ("ta", re.compile(r"[஀-௿]")),  # Tamil
    ("te", re.compile(r"[ఀ-౿]")),  # Telugu
    ("kn", re.compile(r"[ಀ-೿]")),  # Kannada
    ("ml", re.compile(r"[ഀ-ൿ]")),  # Malayalam
]


def detect_language_from_script(text: str) -> str:
    """Returns the language code whose script has the most characters in
    `text`, or "en" if none of the 5 non-Latin scripts appear at all."""
    counts = {code: len(pattern.findall(text)) for code, pattern in _SCRIPT_PATTERNS}
    best_code, best_count = max(counts.items(), key=lambda kv: kv[1])
    return best_code if best_count > 0 else "en"


# Deterministic hazard-term glossary applied BEFORE the LLM translates. The
# LLM mistranslated hazard words on the evaluation set ("लू" -> "Lu", Telugu
# "వడగాలులు" (heat wave) -> "dust storms", Kannada "ಬಿಸಿಗಾಳಿ" -> "hot wind"),
# which sent retrieval to the wrong hazard. Prompt hints fixed Hindi/Kannada
# but not Telugu, so the core hazard words are substituted in code instead.
#
# Keys are word STEMS (final virama/anusvara dropped), because Tamil, Telugu,
# Kannada and Malayalam inflect nouns ("వడగాలుల", "വെള്ളപ്പൊക്കത്തിലെ") and a
# full dictionary form would miss them. Terms were written by the AI
# assistant and are NOT reviewed by native speakers; only longer, unambiguous
# stems are used (short ones like Hindi "आग" occur inside unrelated words).
HAZARD_GLOSSARY = {
    "hi": {"बाढ़": "flood", "चक्रवात": "cyclone", "भूकंप": "earthquake", "सुनामी": "tsunami",
           "भूस्खलन": "landslide", "आकाशीय बिजली": "lightning", "लू": "heat wave"},
    "ta": {"வெள்ள": "flood", "புயல": "cyclone", "நிலநடுக்க": "earthquake", "சுனாமி": "tsunami",
           "நிலச்சரிவ": "landslide", "மின்னல": "lightning", "வெப்ப அலை": "heat wave"},
    "te": {"వరద": "flood", "తుఫాన": "cyclone", "భూకంప": "earthquake", "సునామీ": "tsunami",
           "కొండచరియ": "landslide", "పిడుగ": "lightning", "వడగాల": "heat wave", "మంటల": "fire"},
    "kn": {"ಪ್ರವಾಹ": "flood", "ಚಂಡಮಾರುತ": "cyclone", "ಭೂಕಂಪ": "earthquake", "ಸುನಾಮಿ": "tsunami",
           "ಭೂಕುಸಿತ": "landslide", "ಸಿಡಿಲ": "lightning", "ಬಿಸಿಗಾಳಿ": "heat wave", "ಬೆಂಕಿ": "fire"},
    "ml": {"വെള്ളപ്പൊക്ക": "flood", "ചുഴലിക്കാറ്റ": "cyclone", "ഭൂകമ്പ": "earthquake",
           "സുനാമി": "tsunami", "മണ്ണിടിച്ച": "landslide", "ഇടിമിന്ന": "lightning",
           "ഉഷ്ണതരംഗ": "heat wave", "തീപിടിത്ത": "fire"},
}


def apply_hazard_glossary(text: str, source_language: str) -> str:
    """Replaces known hazard word stems with their English names (spaced),
    leaving the rest for the LLM to translate. Longest stems first so
    compounds win."""
    terms = HAZARD_GLOSSARY.get(source_language)
    if not terms:
        return text
    for term in sorted(terms, key=len, reverse=True):
        text = text.replace(term, f" {terms[term]} ")
    return re.sub(r"\s+", " ", text).strip()


def translate_to_english(text: str, source_language: str) -> str:
    if source_language == "en":
        return text
    text = apply_hazard_glossary(text, source_language)
    language_name = LANGUAGE_NAMES.get(source_language, source_language)
    prompt = _TRANSLATE_PROMPT.format(source_language=language_name, text=text)
    return ollama_client.generate(prompt, max_tokens=160).strip()


def detect_language_and_translate(text: str) -> tuple[str, str]:
    """Detects which supported language the text is written in (via script,
    not the LLM -- see module docstring) and translates it to English.
    Returns (detected_language_code, english_text)."""
    code = detect_language_from_script(text)
    return code, translate_to_english(text, code)
