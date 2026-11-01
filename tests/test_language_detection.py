"""Regression test for a real bug (2026-10-10): asking the LLM to detect
the question's language misidentified Tamil text as Telugu. Unicode script
ranges are disjoint for these 6 languages, so a lookup-based detector is
both more reliable and has zero extra latency."""

from disaster_assistant.generation.translation import detect_language_from_script


def test_detects_english():
    assert detect_language_from_script("What should I do during a flood?") == "en"


def test_detects_hindi():
    assert detect_language_from_script("भूकंप के दौरान मुझे क्या करना चाहिए?") == "hi"


def test_detects_tamil_not_telugu():
    # The exact case that was previously misidentified by the LLM-based detector.
    assert detect_language_from_script("வெள்ளம் வீட்டிற்குள் வந்தால் நான் என்ன செய்ய வேண்டும்?") == "ta"


def test_detects_telugu():
    assert detect_language_from_script("భూకంపం సమయంలో నేను ఏమి చేయాలి?") == "te"


def test_detects_kannada():
    assert detect_language_from_script("ಭೂಕಂಪದ ಸಮಯದಲ್ಲಿ ನಾನು ಏನು ಮಾಡಬೇಕು?") == "kn"


def test_detects_malayalam():
    assert detect_language_from_script("ഭൂകമ്പ സമയത്ത് ഞാൻ എന്ത് ചെയ്യണം?") == "ml"
