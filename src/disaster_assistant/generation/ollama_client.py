"""Thin client for the local Ollama HTTP API -- no cloud calls, localhost only."""

import requests

from disaster_assistant import config


class OllamaError(Exception):
    pass


def generate(prompt: str, timeout_s: int = 120, max_tokens: int | None = None) -> str:
    try:
        return _generate(prompt, timeout_s, max_tokens)
    except OllamaError as e:
        # Greedy decoding can fall into a repetition loop (observed 2026-10-27:
        # a Tamil heat-stroke answer degenerated into a repeated emoji and
        # Ollama aborted with "token repeat limit reached", HTTP 500). Retry
        # once with a repetition penalty -- only in that case, so every other
        # answer stays exactly as evaluated.
        if "repeat limit" not in str(e):
            raise
        return _generate(prompt, timeout_s, max_tokens, extra_options={"repeat_penalty": 1.3})


def _generate(prompt: str, timeout_s: int, max_tokens: int | None, extra_options: dict | None = None) -> str:
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                # gemma4:e4b is a reasoning model that otherwise emits a long
                # internal "thinking" trace before the real answer -- measured
                # directly (2026-10-10): 931 output tokens / ~22s with thinking
                # on, vs. 9 tokens / ~2.5s with it off, for a comparable
                # question. This is the dominant cost in the whole pipeline,
                # not model load or retrieval.
                "think": False,
                # Greedy decoding + fixed seed: with the sampling defaults the
                # same question and the same evidence produced different
                # answers across runs (observed 2026-10-13: identical English
                # search query, different advice per language). For safety
                # guidance that must stay faithful to the evidence, and for
                # experiments to be reproducible, the output should be a
                # function of the prompt only.
                # max_tokens caps the reply length for the short utility calls
                # (YES/NO evidence check, one-line translation) so a chatty
                # model cannot add seconds; answers themselves are uncapped.
                "options": {"temperature": 0, "seed": 0, **({"num_predict": max_tokens} if max_tokens else {}),
                            **(extra_options or {})},
            },
            timeout=timeout_s,
        )
        if resp.status_code >= 400:
            # keep Ollama's own reason (e.g. "token repeat limit reached")
            try:
                reason = resp.json().get("error", "")
            except ValueError:
                reason = resp.text[:200]
            raise OllamaError(f"Ollama returned an error: {resp.status_code} {reason}")
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(f"could not connect to Ollama at {config.OLLAMA_HOST} -- is it running?") from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(f"Ollama did not respond within {timeout_s}s") from e
    except requests.exceptions.HTTPError as e:
        raise OllamaError(f"Ollama returned an error: {e}") from e

    data = resp.json()
    if "response" not in data:
        raise OllamaError(f"unexpected Ollama response shape: {data}")
    return data["response"]
