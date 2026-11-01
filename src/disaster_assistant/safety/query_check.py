"""Detects queries too vague to retrieve against safely.

Found 2026-10-13 (scripts/stress_api.py): the one-word message "help" was
answered with fire-alarm evacuation steps -- real NDMA guidance, but for a
hazard the user never mentioned. The evidence check cannot catch this
because the passages are genuine guidance; the problem is the QUESTION
carries no information. Rather than guess a hazard in an emergency, ask.

Operates on the English search query (the translator's output for
non-English input).
"""

import difflib

from disaster_assistant.knowledge.indexing import lexical_tokens

# Stemmed hazard words (lexical_tokens strips plurals). A one-word query that
# names a hazard ("flood", "earthquake") is specific enough to retrieve for.
_HAZARD_WORDS = frozenset(
    """flood cyclone hurricane typhoon earthquake quake tsunami landslide lightning thunderstorm
    fire heatwave heat drought storm avalanche chemical radiation""".split()
)


def _names_a_hazard(token: str) -> bool:
    # Fuzzy, because transcripts and typed text contain misspellings
    # ("earthqake", "flud"): a strict match wrongly flagged
    # "what to do in an earthqake" as vague (stress test, 2026-10-13).
    return token in _HAZARD_WORDS or bool(difflib.get_close_matches(token, _HAZARD_WORDS, n=1, cutoff=0.8))


def is_too_vague(search_question: str) -> bool:
    tokens = lexical_tokens(search_question)
    if any(_names_a_hazard(t) for t in tokens):
        return False
    return len(tokens) < 2


CLARIFY_MESSAGE = {
    "en": (
        "I need a little more detail to help. Please say what is happening, for example: flood, "
        "cyclone, earthquake or fire. If you are in immediate danger, call 112, 1070 or 1077."
    ),
    # hi/ta/te/kn/ml: machine translations by the AI assistant, NOT reviewed
    # by native speakers (same caveat as the abstention messages).
    "hi": (
        "मदद के लिए मुझे थोड़ी और जानकारी चाहिए। कृपया बताएँ कि क्या हो रहा है, जैसे: बाढ़, चक्रवात, "
        "भूकंप या आग। अगर आप तुरंत खतरे में हैं, तो 112, 1070 या 1077 पर कॉल करें।"
    ),
    "ta": (
        "உதவ எனக்கு இன்னும் கொஞ்சம் விவரம் தேவை. என்ன நடக்கிறது என்று சொல்லுங்கள், எடுத்துக்காட்டாக: "
        "வெள்ளம், புயல், நிலநடுக்கம் அல்லது தீ. உடனடி ஆபத்தில் இருந்தால் 112, 1070 அல்லது 1077 ஐ அழைக்கவும்."
    ),
    "te": (
        "సహాయం చేయడానికి నాకు ఇంకొంచెం వివరాలు కావాలి. ఏమి జరుగుతోందో చెప్పండి, ఉదాహరణకు: వరద, తుఫాను, "
        "భూకంపం లేదా అగ్నిప్రమాదం. మీరు వెంటనే ప్రమాదంలో ఉంటే 112, 1070 లేదా 1077కి కాల్ చేయండి."
    ),
    "kn": (
        "ಸಹಾಯ ಮಾಡಲು ನನಗೆ ಇನ್ನಷ್ಟು ವಿವರಗಳು ಬೇಕು. ಏನು ನಡೆಯುತ್ತಿದೆ ಎಂದು ತಿಳಿಸಿ, ಉದಾಹರಣೆಗೆ: ಪ್ರವಾಹ, ಚಂಡಮಾರುತ, "
        "ಭೂಕಂಪ ಅಥವಾ ಬೆಂಕಿ. ನೀವು ತಕ್ಷಣದ ಅಪಾಯದಲ್ಲಿದ್ದರೆ 112, 1070 ಅಥವಾ 1077 ಗೆ ಕರೆ ಮಾಡಿ."
    ),
    "ml": (
        "സഹായിക്കാൻ എനിക്ക് കുറച്ചുകൂടി വിവരങ്ങൾ വേണം. എന്താണ് സംഭവിക്കുന്നതെന്ന് പറയുക, ഉദാഹരണത്തിന്: "
        "വെള്ളപ്പൊക്കം, ചുഴലിക്കാറ്റ്, ഭൂകമ്പം അല്ലെങ്കിൽ തീപിടിത്തം. നിങ്ങൾ ഉടനടി അപകടത്തിലാണെങ്കിൽ "
        "112, 1070 അല്ലെങ്കിൽ 1077 ൽ വിളിക്കുക."
    ),
}
