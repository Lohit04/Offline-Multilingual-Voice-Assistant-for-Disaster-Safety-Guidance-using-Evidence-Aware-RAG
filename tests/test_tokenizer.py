"""Regression tests for two real bugs found 2026-10-10:

1. The original tokenizer was ASCII-only ([a-z0-9]+) and silently produced
   zero tokens for any non-Latin script, so BM25 could never match a Tamil
   query against anything, regardless of transcription quality.
2. The first fix (a Unicode \\w+ regex) was *also* wrong for Tamil: Tamil
   words are base letters + combining vowel-sign/virama codepoints, and
   those combining marks aren't in Python re's \\w, so the regex fragmented
   every word at each combining mark instead of keeping it as one token.
   Splitting on whitespace and stripping edge punctuation avoids this,
   since it never looks inside a non-space run.
"""

from disaster_assistant.knowledge.indexing import tokenize


def test_tokenizes_english():
    assert tokenize("Flood Safety Guidelines 2008") == ["flood", "safety", "guidelines", "2008"]


def test_tokenizes_tamil_as_whole_words_not_fragments():
    tokens = tokenize("வெள்ளம் வீட்டிற்குள் வந்தால்")
    assert tokens == ["வெள்ளம்", "வீட்டிற்குள்", "வந்தால்"]


def test_strips_edge_punctuation_but_keeps_internal_characters():
    assert tokenize("don't, enter!") == ["don't", "enter"]
