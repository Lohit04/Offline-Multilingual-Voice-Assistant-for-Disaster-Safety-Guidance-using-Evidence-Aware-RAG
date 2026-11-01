from disaster_assistant.safety.live_info import is_live_info_query, apply_live_info_guard


def test_detects_live_info_phrasings():
    assert is_live_info_query("Where is the nearest open relief shelter right now?")
    assert is_live_info_query("Is the highway currently open?")
    assert is_live_info_query("What is the current road status?")


def test_does_not_flag_general_safety_questions():
    assert not is_live_info_query("What should I do if floodwater enters my house?")
    assert not is_live_info_query("What should I do during an earthquake?")


def test_guard_prepends_disclaimer_only_when_triggered():
    guarded = apply_live_info_guard("some answer", "where is the nearest open shelter right now", "en")
    assert guarded.startswith("Note:")
    assert "some answer" in guarded

    unguarded = apply_live_info_guard("some answer", "what should I do during a flood", "en")
    assert unguarded == "some answer"


def test_guard_falls_back_to_english_for_unknown_language():
    guarded = apply_live_info_guard("ans", "what is open right now", "fr")
    assert guarded.startswith("Note:")


def test_detects_translation_variants_seen_in_multilingual_eval():
    # Real English queries the translator produced from Hindi/Tamil (2026-10-13)
    assert is_live_info_query("Is the road in Chennai still closed due to flooding?")
    assert is_live_info_query("Has an order been given to evacuate my village today?")
    assert is_live_info_query("Is my village ordered to be evacuated today?")
    assert is_live_info_query("Will it rain in Chennai tomorrow?")


def test_widened_patterns_keep_core_safety_questions_clean():
    assert not is_live_info_query("How should I prepare for a cyclone?")
    assert not is_live_info_query("How should I prepare to evacuate before a flood?")
    assert not is_live_info_query("Is it safe to walk or drive through flood water?")
    assert not is_live_info_query("What should I do after an earthquake stops?")


def test_detects_phrasings_missed_in_heldout_eval():
    # English search queries the translator produced on the held-out set (2026-10-13)
    assert is_live_info_query("Is there a cyclone warning for Chennai now?")
    assert is_live_info_query("How many people have been rescued so far in the current flood?")
    assert is_live_info_query("How many people have been rescued in the current flood?")
    assert is_live_info_query("Is a cyclone warning issued for Chennai right now?")


def test_core_safety_questions_still_clean_after_widening():
    for q in [
        "What should I do if I am driving when an earthquake strikes?",
        "How can I protect my house from strong cyclone winds?",
        "What should I do during a chemical accident?",
        "What does a cyclone warning mean?",
        "How can I stay safe during a cold wave?",
        "What should I do after a cyclone has passed?",
    ]:
        assert not is_live_info_query(q), q
