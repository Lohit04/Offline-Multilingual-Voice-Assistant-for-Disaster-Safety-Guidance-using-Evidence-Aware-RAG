"""Regression for the stress-test finding: "help" was answered with fire-alarm
instructions. Vague queries must be clarified, not guessed at."""

from disaster_assistant.safety.query_check import is_too_vague, CLARIFY_MESSAGE

# Numbers shown to users must be the verified ones from documents.json
VERIFIED = {"112", "1070", "1077"}


def test_bare_help_is_too_vague():
    assert is_too_vague("help")
    assert is_too_vague("Help!")
    assert is_too_vague("what do I do")


def test_single_hazard_word_is_specific_enough():
    assert not is_too_vague("flood")
    assert not is_too_vague("Earthquake")
    assert not is_too_vague("fire!")


def test_misspelled_hazard_word_is_still_specific():
    # Regression from the stress test: typo made this look vague
    assert not is_too_vague("what to do in an earthqake")
    assert not is_too_vague("floood")
    assert not is_too_vague("cyclon")


def test_similar_looking_non_hazards_stay_vague():
    assert is_too_vague("help")
    assert is_too_vague("hello")
    assert is_too_vague("please")


def test_normal_questions_are_not_vague():
    assert not is_too_vague("What should I do if floodwater enters my house?")
    assert not is_too_vague("water entering my house")


def test_clarify_messages_exist_for_all_languages_and_carry_verified_numbers():
    assert set(CLARIFY_MESSAGE) == {"en", "hi", "ta", "te", "kn", "ml"}
    for lang, msg in CLARIFY_MESSAGE.items():
        for n in VERIFIED:
            assert n in msg, (lang, n)
