"""Hazard words are substituted in code before LLM translation, because the
LLM mistranslated them (Telugu heat wave -> "dust storms", Hindi लू -> "Lu")."""

from disaster_assistant.generation.translation import apply_hazard_glossary, HAZARD_GLOSSARY


def test_telugu_heat_wave_is_substituted():
    out = apply_hazard_glossary("వడగాలుల నుండి నన్ను నేను ఎలా రక్షించుకోవాలి?", "te")
    assert "heat wave" in out


def test_hindi_loo_becomes_heat_wave():
    assert "heat wave" in apply_hazard_glossary("मैं लू से खुद को कैसे बचा सकता हूँ?", "hi")


def test_all_core_hazards_present_for_every_non_english_language():
    for lang in ("hi", "ta", "te", "kn", "ml"):
        values = set(HAZARD_GLOSSARY[lang].values())
        assert {"flood", "cyclone", "earthquake", "tsunami", "landslide", "lightning", "heat wave"} <= values, lang


def test_every_evaluation_hazard_question_gets_a_substitution():
    # The glossary must actually fire on the real evaluation questions
    import json
    from disaster_assistant import config

    data = json.loads((config.ROOT / "data" / "evaluation" / "questions_v1.json").read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in data["items"]}
    for qid, english_word in [("A01", "flood"), ("A02", "earthquake"), ("A07", "tsunami"), ("A08", "heat wave")]:
        for lang in ("hi", "ta", "te", "kn", "ml"):
            out = apply_hazard_glossary(by_id[qid]["q"][lang], lang)
            assert english_word in out, (qid, lang, out)


def test_unknown_language_and_english_are_untouched():
    assert apply_hazard_glossary("flood", "en") == "flood"
    assert apply_hazard_glossary("anything", "fr") == "anything"


def test_text_without_hazard_words_is_unchanged():
    q = "நேற்றைய கிரிக்கெட் போட்டியில் யார் வென்றார்?"
    assert apply_hazard_glossary(q, "ta") == q
