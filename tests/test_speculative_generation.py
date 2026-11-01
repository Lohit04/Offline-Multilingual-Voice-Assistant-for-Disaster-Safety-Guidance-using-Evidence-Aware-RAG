"""Answers are generated speculatively while the evidence check runs. The
safety property that must survive that optimisation: if the check says NO, the
drafted answer is NEVER returned -- the user gets the abstention message."""

from disaster_assistant import config
from disaster_assistant.generation import answer as answer_mod
from disaster_assistant.generation import ollama_client
from disaster_assistant.generation.answer import ABSTENTION_MESSAGE, answer_question
from disaster_assistant.retrieval.hybrid import RetrievedChunk


class _FakeRetriever:
    def search(self, query, top_k=5):
        return [RetrievedChunk("c1", "doc", "flood", 1, "Move to higher ground.", 5.0, 0.9, 0.9)]


def _fake_llm(check_says):
    def fake_generate(prompt, timeout_s=120, max_tokens=None):
        if "Do the passages contain" in prompt or "can help with a question" in prompt:
            return check_says
        return "DRAFT ANSWER"
    return fake_generate


def test_rejected_draft_is_never_returned(monkeypatch):
    monkeypatch.setattr(ollama_client, "generate", _fake_llm("NO"))
    ans = answer_question(_FakeRetriever(), "How do I stay safe in a flood?", language="en")
    assert ans.abstained and ans.abstention_reason == "llm_evidence_check_no"
    assert "DRAFT" not in ans.text
    assert ans.text == ABSTENTION_MESSAGE["en"]
    assert ans.sources == []


def test_approved_draft_is_returned(monkeypatch):
    monkeypatch.setattr(ollama_client, "generate", _fake_llm("YES"))
    ans = answer_question(_FakeRetriever(), "How do I stay safe in a flood?", language="en")
    assert not ans.abstained and "DRAFT ANSWER" in ans.text
    assert "evidence_check" in ans.timings and "generate" in ans.timings


def test_same_decision_with_speculation_disabled(monkeypatch):
    monkeypatch.setattr(config, "SPECULATIVE_GENERATION", False)
    monkeypatch.setattr(ollama_client, "generate", _fake_llm("NO"))
    assert answer_question(_FakeRetriever(), "How do I stay safe in a flood?", language="en").abstained
    monkeypatch.setattr(ollama_client, "generate", _fake_llm("YES"))
    assert not answer_question(_FakeRetriever(), "How do I stay safe in a flood?", language="en").abstained


def test_weak_retrieval_never_starts_generation(monkeypatch):
    calls = []

    class _Weak:
        def search(self, query, top_k=5):
            return [RetrievedChunk("c1", "doc", "flood", 1, "x", 0.0, 0.3, 0.1)]

    monkeypatch.setattr(ollama_client, "generate", lambda *a, **k: calls.append(1) or "x")
    ans = answer_question(_Weak(), "How do I stay safe in a flood?", language="en")
    assert ans.abstained and calls == []
