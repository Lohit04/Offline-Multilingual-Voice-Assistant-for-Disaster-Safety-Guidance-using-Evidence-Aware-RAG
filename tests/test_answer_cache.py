"""Instant-answer cache: the safety rules that let it skip the LLM.

1. A question naming a hazard (any language) may only get that hazard's
   prepared answer -- never another hazard's, however similar the wording.
2. Questions about current conditions ("now", "today"; English patterns or
   Indian-language time words) never get a prepared answer.
3. Below the confidence bar, no instant answer -- the live pipeline runs.
"""

import numpy as np

from disaster_assistant.retrieval import answer_cache as ac
from disaster_assistant.safety.live_info import mentions_current_time


def test_hazard_detected_in_english_and_indian_languages():
    assert "flood" in ac.question_hazards("What should I do if floodwater enters my house?")
    assert ac.question_hazards("வெள்ளம் வீட்டிற்குள் வந்தால் என்ன செய்வது?") >= {"flood"}
    assert ac.question_hazards("भूकंप के दौरान क्या करें?") == {"earthquake"}
    assert ac.question_hazards("ಭೂಕಂಪದ ಸಮಯದಲ್ಲಿ ಏನು ಮಾಡಬೇಕು?") == {"earthquake"}
    assert ac.question_hazards("How do I stay safe?") == set()


def test_current_time_questions_are_flagged_in_every_language():
    assert mentions_current_time("Which shelters are open right now?")
    assert mentions_current_time("क्या अभी चेन्नई के लिए चक्रवात की चेतावनी जारी है?")
    assert mentions_current_time("இன்று வெள்ளம் வருமா?")
    assert not mentions_current_time("What should I do during a cyclone?")


def _tiny_cache():
    """Two situations whose phrasings embed almost identically, so only the
    hazard restriction can tell them apart."""
    cache = ac.AnswerCache.__new__(ac.AnswerCache)
    cache.intents = [{"id": "flood.during", "hazard": "flood", "answers": {"en": "flood answer"}},
                     {"id": "earthquake.during", "hazard": "earthquake", "answers": {"en": "quake answer"}}]
    cache.texts = ["What should I do during a flood?", "What should I do during an earthquake?"]
    cache.owner = np.array([0, 1])
    cache.hazard_of = np.array(["flood", "earthquake"])
    v = np.array([1.0, 0.0], dtype="float32")
    cache.emb = np.stack([v, v + np.array([0.0, 0.001], dtype="float32")])
    cache.rej_emb = np.zeros((0, 2), dtype="float32")
    return cache


def test_question_closer_to_a_reject_phrasing_gets_no_instant_answer():
    cache = _tiny_cache()
    cache.rej_emb = np.array([[1.0, 0.0]], dtype="float32")  # e.g. "What causes floods?"
    assert cache.match("What causes floods?", np.array([1.0, 0.0], dtype="float32")) is None


def test_named_hazard_restricts_the_match_even_if_another_is_closer():
    cache = _tiny_cache()
    q = np.array([0.0, 1.0], dtype="float32")  # slightly closer to the earthquake phrasing
    q = q + np.array([1.0, 0.0], dtype="float32")
    q /= np.linalg.norm(q)
    m = cache.match("What should I do during a flood?", q)
    assert m is None or m.intent["hazard"] == "flood"


def test_weak_match_falls_through_to_live_pipeline():
    cache = _tiny_cache()
    unrelated = np.array([0.0, 1.0], dtype="float32")  # cosine ~0 with both phrasings
    assert cache.match("How do I file an insurance claim?", unrelated) is None
