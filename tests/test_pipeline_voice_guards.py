"""Voice pipeline guards found by the 2026-10-13 stress test: silence was
transcribed as a Norwegian subtitle credit and then answered. Unsupported
detected languages must be rejected, not silently answered in English."""

import pytest

from disaster_assistant import pipeline
from disaster_assistant.audio.asr import Transcription


def test_unsupported_detected_language_is_rejected(monkeypatch):
    monkeypatch.setattr(
        pipeline, "transcribe", lambda path, language=None: Transcription("Teksting av Nicolai Winther", "nn", 0.6)
    )
    with pytest.raises(pipeline.EmptyTranscriptionError, match="supported language"):
        pipeline.process_voice_question(object(), "in.wav", "out.wav")


def test_empty_transcription_is_rejected(monkeypatch):
    monkeypatch.setattr(pipeline, "transcribe", lambda path, language=None: Transcription("   ", "en", 0.9))
    with pytest.raises(pipeline.EmptyTranscriptionError):
        pipeline.process_voice_question(object(), "in.wav", "out.wav")
