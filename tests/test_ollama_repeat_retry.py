"""Greedy decoding can loop on one token; Ollama then aborts with HTTP 500
"token repeat limit reached" (observed 2026-10-27 on a Tamil heat-stroke
answer). The client must retry once with a repetition penalty -- and only
for that error, so normal answers are generated exactly as evaluated."""

from unittest import mock

import pytest

from disaster_assistant.generation import ollama_client


class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def test_repeat_limit_error_is_retried_with_a_repetition_penalty():
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json["options"])
        if len(calls) == 1:
            return FakeResp(500, {"error": "prediction aborted, token repeat limit reached"})
        return FakeResp(200, {"response": "1. Move the person to a cool place."})

    with mock.patch.object(ollama_client.requests, "post", fake_post):
        assert ollama_client.generate("prompt") == "1. Move the person to a cool place."
    assert "repeat_penalty" not in calls[0]          # first attempt unchanged
    assert calls[1]["repeat_penalty"] == 1.3          # retry adds the penalty
    assert calls[1]["temperature"] == 0               # still deterministic


def test_other_server_errors_are_not_retried():
    calls = []

    def fake_post(url, json, timeout):
        calls.append(1)
        return FakeResp(500, {"error": "model not found"})

    with mock.patch.object(ollama_client.requests, "post", fake_post):
        with pytest.raises(ollama_client.OllamaError, match="model not found"):
            ollama_client.generate("prompt")
    assert len(calls) == 1
