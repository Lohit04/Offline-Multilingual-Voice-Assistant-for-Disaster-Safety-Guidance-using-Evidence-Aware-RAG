"""large-v3 wrote spoken Malayalam in Gurmukhi script (2026-10-13). transcribe()
retries once with an in-language prompt when the output script does not match
the detected language. Stubbed model: no GPU or audio needed."""

from types import SimpleNamespace

from disaster_assistant.audio import asr


def _stub(monkeypatch, first_text, retry_text, lang="ml"):
    calls = []

    def fake_run(model, audio_path, **kwargs):
        calls.append(kwargs)
        text = retry_text if kwargs.get("initial_prompt") else first_text
        return text, SimpleNamespace(language=lang, language_probability=0.9)

    monkeypatch.setattr(asr, "get_model", lambda: object())
    monkeypatch.setattr(asr, "_run", fake_run)
    return calls


def test_retries_with_prompt_when_script_is_wrong(monkeypatch):
    calls = _stub(monkeypatch, first_text="ਭੂਕਂਪ ਸਮੇਤੁ ਨਾਨ", retry_text="ഭൂകമ്പ സമയത്ത് ഞാൻ")
    result = asr.transcribe("x.wav")
    assert len(calls) == 2 and calls[1]["language"] == "ml" and calls[1]["initial_prompt"]
    assert result.text == "ഭൂകമ്പ സമയത്ത് ഞാൻ" and result.retried_for_script


def test_no_retry_when_script_already_matches(monkeypatch):
    calls = _stub(monkeypatch, first_text="ഭൂകമ്പ സമയത്ത്", retry_text="unused")
    result = asr.transcribe("x.wav")
    assert len(calls) == 1 and not result.retried_for_script


def test_keeps_first_pass_if_retry_is_no_better(monkeypatch):
    _stub(monkeypatch, first_text="ਭੂਕਂਪ", retry_text="still not malayalam")
    result = asr.transcribe("x.wav")
    assert result.text == "ਭੂਕਂਪ" and not result.retried_for_script


def test_english_is_never_retried(monkeypatch):
    calls = _stub(monkeypatch, first_text="what to do in an earthquake", retry_text="unused", lang="en")
    asr.transcribe("x.wav")
    assert len(calls) == 1
