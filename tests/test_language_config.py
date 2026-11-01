"""Guards against the two per-language config tables (which languages the
prompt can name, and which TTS engine/voice handles each) silently drifting
apart -- e.g. adding a language to one without the other."""

from disaster_assistant import config
from disaster_assistant.generation.prompts import LANGUAGE_NAMES
from disaster_assistant.generation.answer import ABSTENTION_MESSAGE


def test_every_named_language_has_a_tts_voice():
    missing = set(LANGUAGE_NAMES) - set(config.TTS_CONFIG)
    assert not missing, f"languages with a prompt name but no TTS voice: {missing}"


def test_every_named_language_has_an_abstention_message():
    missing = set(LANGUAGE_NAMES) - set(ABSTENTION_MESSAGE)
    assert not missing, f"languages with a prompt name but no abstention message: {missing}"


def test_tts_config_entries_are_well_formed():
    for lang, voice_config in config.TTS_CONFIG.items():
        assert voice_config["engine"] in ("piper", "mms"), f"{lang}: unknown engine {voice_config['engine']!r}"
        if voice_config["engine"] == "piper":
            assert "voice" in voice_config
        else:
            assert "model" in voice_config
