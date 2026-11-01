"""Generation must be a function of the prompt only (greedy, fixed seed).
Observed 2026-10-13: with Ollama's default sampling, the same English search
query and evidence produced different advice across runs/languages."""

from disaster_assistant.generation import ollama_client


class _FakeResp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "ok"}


def test_generate_uses_greedy_decoding_and_fixed_seed(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return _FakeResp()

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    assert ollama_client.generate("hello") == "ok"
    opts = captured["json"]["options"]
    assert opts["temperature"] == 0
    assert "seed" in opts


def test_max_tokens_caps_reply_only_when_requested(monkeypatch):
    captured = []

    def fake_post(url, json=None, timeout=None):
        captured.append(json)
        return _FakeResp()

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    ollama_client.generate("a")
    ollama_client.generate("b", max_tokens=6)
    assert "num_predict" not in captured[0]["options"]
    assert captured[1]["options"]["num_predict"] == 6
