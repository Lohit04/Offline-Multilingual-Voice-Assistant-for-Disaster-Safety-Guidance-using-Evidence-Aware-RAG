"""Smoke tests for the FastAPI backend (the web/mobile-browser frontend
talks to this). Uses the real indexes/Ollama -- not a pure unit test, but
these are cheap and catch wiring mistakes the other tests can't (route
paths, form-field names, response shape) before a phone ever hits them."""

from fastapi.testclient import TestClient

from disaster_assistant.api.server import app

client = TestClient(app)


def test_index_serves_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Disaster Safety Assistant" in resp.content


def test_static_app_js_is_served():
    resp = client.get("/static/app.js")
    assert resp.status_code == 200


def test_ask_out_of_scope_question_abstains():
    resp = client.post("/api/ask", data={"question": "What is the capital of France?", "language": "en"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is True


def test_ask_answerable_question_returns_sources():
    resp = client.post(
        "/api/ask",
        data={"question": "What should I do if floodwater enters my house?", "language": "en"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is False
    assert len(data["sources"]) > 0


def test_index_has_helplines_and_example_chips():
    html = client.get("/").text
    for number in ("tel:112", "tel:1070", "tel:1077"):
        assert number in html
    assert 'id="example-chips"' in html


def test_helpline_numbers_in_ui_match_verified_registry():
    import json
    from disaster_assistant import config

    registry = json.loads(config.DOCUMENTS_REGISTRY.read_text(encoding="utf-8"))
    verified = {c["phone"] for c in registry["verified_contacts"]}
    html = client.get("/").text
    shown = {n for n in ("112", "1070", "1077") if f"tel:{n}" in html}
    assert shown <= verified


def test_health_reports_ollama_state(monkeypatch):
    from disaster_assistant.api import server

    class _Tags:
        def json(self):
            return {"models": [{"name": server.config.OLLAMA_MODEL}]}

    monkeypatch.setattr(server.requests, "get", lambda *a, **k: _Tags())
    ok = client.get("/api/health").json()
    assert ok["server"] == "ok" and ok["ollama"] is True and ok["model_present"] is True

    def boom(*a, **k):
        raise server.requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(server.requests, "get", boom)
    down = client.get("/api/health").json()
    assert down["server"] == "ok" and down["ollama"] is False and down["model_present"] is False


def test_tts_rejects_empty_text_and_unknown_language():
    assert client.post("/api/tts", data={"text": "   ", "language": "en"}).status_code == 400
    assert client.post("/api/tts", data={"text": "hello", "language": "xx"}).status_code == 400


def test_tts_returns_playable_audio_for_english():
    import base64

    resp = client.post("/api/tts", data={"text": "Move to higher ground.", "language": "en"})
    assert resp.status_code == 200
    wav = base64.b64decode(resp.json()["audio_base64"])
    assert wav[:4] == b"RIFF" and len(wav) > 2000


def test_instant_answer_by_situation_id():
    ids = client.get("/api/instant_ids").json()["ids"]
    assert "flood.during" in ids
    d = client.get("/api/instant", params={"id": "flood.during", "language": "ta"}).json()
    assert d["cached"] and d["answer_language"] == "ta" and d["sources"]
    assert client.get("/api/instant", params={"id": "no.such"}).status_code == 404
