"""Detect questions asking for live/real-time information the system cannot
possibly answer from static documents (current shelter status, road
conditions, live weather, active evacuation orders, etc).

This exists because the evidence-sufficiency gate alone is not enough:
testing (2026-10-10, see docs/implementation_status.md finding #2) showed a
question like "where is the nearest open relief shelter right now" passes
sufficiency (the documents genuinely discuss shelter-siting policy) and the
only thing stopping an over-confident answer was the LLM choosing to follow
a prompt instruction -- not a structural guarantee. This module adds a
structural one: when a live-info question is detected, a fixed disclaimer
is prepended to the final answer in code, not left to the model.

English-only detection for now. Rule-based keyword matching is a coarse
heuristic (false negatives are expected for phrasings not in the list below;
no false-positive/negative rate has been measured), not a validated
classifier -- flagged here rather than left implicit.
"""

import re

_LIVE_INFO_PATTERNS = [
    r"\bright now\b",
    r"\bcurrently\b",
    r"\bat this (moment|time)\b",
    r"\b(is|are)\s+\w+\s+open\b",
    r"\bnearest (open|available)\b",
    r"\bcurrent (location|status|condition|road|weather|situation)\b",
    r"\bavailable (now|today)\b",
    r"\blive (update|status)\b",
    r"\bhas .* been evacuated\b",
    r"\bwhich (shelters?|camps?|roads?) (is|are) (open|closed|blocked|available)\b",
    # Added 2026-10-13 after the multilingual evaluation: the English text the
    # translator produces varies ("still closed", "today", "ordered to be
    # evacuated"), and the original phrasings missed those, so a live question
    # got an unqualified answer. A false positive only adds a disclaimer.
    r"\b(today|tonight|tomorrow)\b",
    r"\bstill (closed|open|blocked|flooded|raining|available)\b",
    r"\border(s|ed)?\b.{0,40}\bevacuat",
    r"\bevacuat\w*\b.{0,40}\b(order|ordered|announced|issued)\b",
    r"\b(road|roads|bridge|route|highway)\b.{0,40}\b(closed|blocked|open|flooded|washed)\b",
    r"\bnearest (shelter|camp|hospital|relief)\b",
    r"\b(will it rain|forecast|weather in)\b",
    # Added after the held-out evaluation (2026-10-13): these phrasings were
    # missed -- "current flood", "so far", "warning ... now", casualty counts.
    r"\bso far\b",
    r"\bcurrent (flood|cyclone|earthquake|disaster|storm|emergency|alert|warning)s?\b",
    r"\b(warning|alert)s?\b.{0,40}\b(now|today|currently|issued|in effect|active)\b",
    r"\b(is|are) there (a|an|any) .{0,30}(warning|alert|evacuation)\b",
    r"\b(how many|number of)\b.{0,50}\b(rescued|killed|affected|evacuated|died|injured|missing|dead)\b",
    r"\b(is|are|has|have)\b.{0,50}\b(now|at the moment|at present)\b",
]
_LIVE_INFO_RE = re.compile("|".join(_LIVE_INFO_PATTERNS), re.IGNORECASE)

# Machine-assisted translation, NOT verified by a native Tamil speaker --
# per CLAUDE.md SS12, must not be presented as an official/verified
# translation. Flag for review before this is relied on in any real
# deployment or demo to Tamil speakers.
LIVE_INFO_DISCLAIMER = {
    "en": (
        "Note: I cannot confirm current, real-time conditions (shelter availability, "
        "road status, active evacuation orders, live weather) from these static documents. "
        "Please verify with official local sources before acting on time-sensitive details below."
    ),
    "ta": (
        "குறிப்பு (இயந்திர மொழிபெயர்ப்பு, சரிபார்க்கப்படவில்லை): தற்போதைய நிகழ்நேர தகவல்களை "
        "(முகாம் கிடைக்குமா, சாலை நிலை, வெளியேற்ற உத்தரவுகள், வானிலை) இந்த நிலையான ஆவணங்களில் "
        "இருந்து உறுதிப்படுத்த முடியாது. முடிவு எடுப்பதற்கு முன் உள்ளூர் அதிகாரிகளை தொடர்பு கொள்ளவும்."
    ),
    # hi/te/kn/ml added 2026-10-13: previously these languages silently fell
    # back to the English text above. Same caveat as Tamil -- machine
    # translations by the AI assistant, NOT reviewed by native speakers.
    "hi": (
        "नोट (मशीनी अनुवाद, सत्यापित नहीं): मैं इन स्थिर दस्तावेज़ों से वर्तमान, रीयल-टाइम स्थितियों "
        "(शिविर की उपलब्धता, सड़क की स्थिति, निकासी आदेश, मौसम) की पुष्टि नहीं कर सकता। "
        "कार्रवाई करने से पहले कृपया स्थानीय अधिकारियों से संपर्क करें।"
    ),
    "te": (
        "గమనిక (యంత్ర అనువాదం, ధృవీకరించబడలేదు): ఈ స్థిర పత్రాల నుండి ప్రస్తుత, రియల్-టైమ్ పరిస్థితులను "
        "(శిబిరాల లభ్యత, రహదారి స్థితి, ఖాళీ చేయించే ఆదేశాలు, వాతావరణం) నేను నిర్ధారించలేను. "
        "చర్య తీసుకునే ముందు దయచేసి స్థానిక అధికారులను సంప్రదించండి."
    ),
    "kn": (
        "ಗಮನಿಸಿ (ಯಂತ್ರ ಅನುವಾದ, ಪರಿಶೀಲಿಸಲಾಗಿಲ್ಲ): ಈ ಸ್ಥಿರ ದಾಖಲೆಗಳಿಂದ ಪ್ರಸ್ತುತ, ನೈಜ-ಸಮಯದ ಪರಿಸ್ಥಿತಿಗಳನ್ನು "
        "(ಶಿಬಿರ ಲಭ್ಯತೆ, ರಸ್ತೆ ಸ್ಥಿತಿ, ಸ್ಥಳಾಂತರ ಆದೇಶಗಳು, ಹವಾಮಾನ) ನಾನು ದೃಢೀಕರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ. "
        "ಕ್ರಮ ಕೈಗೊಳ್ಳುವ ಮೊದಲು ದಯವಿಟ್ಟು ಸ್ಥಳೀಯ ಅಧಿಕಾರಿಗಳನ್ನು ಸಂಪರ್ಕಿಸಿ."
    ),
    "ml": (
        "ശ്രദ്ധിക്കുക (മെഷീൻ വിവർത്തനം, പരിശോധിച്ചിട്ടില്ല): ഈ സ്ഥിര രേഖകളിൽ നിന്ന് നിലവിലെ തത്സമയ സാഹചര്യങ്ങൾ "
        "(ക്യാമ്പ് ലഭ്യത, റോഡ് നില, ഒഴിപ്പിക്കൽ ഉത്തരവുകൾ, കാലാവസ്ഥ) സ്ഥിരീകരിക്കാൻ എനിക്ക് കഴിയില്ല. "
        "നടപടിയെടുക്കുന്നതിന് മുമ്പ് ദയവായി പ്രാദേശിക അധികാരികളെ ബന്ധപ്പെടുക."
    ),
}


def is_live_info_query(text: str) -> bool:
    return bool(_LIVE_INFO_RE.search(text))


# "now / today / currently" in each Indian language. Used ONLY to keep such
# questions off the instant-answer path (retrieval/answer_cache.py), which
# has no translation step for the English patterns above to act on -- they
# then go through the live pipeline, where translation + the English guard
# apply. A false positive only makes that answer slower. Added 2026-10-26
# after calibration showed a Hindi "is there a cyclone warning NOW" question
# scoring just under the instant-answer threshold. Common words chosen by
# the AI assistant, not reviewed by native speakers.
LIVE_TIME_WORDS = {
    "hi": ["अभी", "आज", "वर्तमान", "फिलहाल", "इस समय"],
    "ta": ["இப்போது", "இன்று", "தற்போது", "இப்ப", "இந்த நேரத்தில்"],
    "te": ["ఇప్పుడు", "ఈరోజు", "ఈ రోజు", "ప్రస్తుతం", "ఈ సమయంలో"],
    "kn": ["ಈಗ", "ಇಂದು", "ಪ್ರಸ್ತುತ", "ಈ ಸಮಯದಲ್ಲಿ"],
    "ml": ["ഇപ്പോൾ", "ഇന്ന്", "നിലവിൽ", "ഈ സമയത്ത്"],
}


def mentions_current_time(text: str) -> bool:
    """English live-info patterns, or a now/today word in an Indian language."""
    return is_live_info_query(text) or any(w in text for words in LIVE_TIME_WORDS.values() for w in words)


def apply_live_info_guard(answer_text: str, question: str, language: str = "en") -> str:
    if not is_live_info_query(question):
        return answer_text
    disclaimer = LIVE_INFO_DISCLAIMER.get(language, LIVE_INFO_DISCLAIMER["en"])
    return f"{disclaimer}\n\n{answer_text}"
