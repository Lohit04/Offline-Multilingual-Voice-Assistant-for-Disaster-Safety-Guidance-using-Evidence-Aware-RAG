"""Sanity check that the LLM endpoint is pinned to localhost -- this is the
one thing in the pipeline that could silently become a cloud dependency if
someone changed config.OLLAMA_HOST, which would break the project's core
offline-operation requirement without any other test catching it."""

from disaster_assistant import config


def test_ollama_host_is_local():
    assert "localhost" in config.OLLAMA_HOST or "127.0.0.1" in config.OLLAMA_HOST
