"""Tests for the first-stage evidence gate (safety/abstention.py).

History: the original gate required the TOP result to share keywords with the
query (BM25 > 0). That was removed 2026-10-13 -- it wrongly refused
answerable questions whose best evidence was a dense-only match (dense
0.88-0.895, zero keyword overlap) in the multilingual evaluation. The
semantic decision now belongs to safety/evidence_check.py; this gate only
rejects questions nothing in the corpus resembles, using the BEST dense score
among the returned chunks (not necessarily results[0]).
"""

from disaster_assistant.retrieval.hybrid import RetrievedChunk
from disaster_assistant.safety.abstention import assess, DENSE_SCORE_FLOOR


def _chunk(dense_score: float, bm25_score: float, cid: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="doc1",
        disaster_category="flood",
        page=1,
        text="some evidence text",
        bm25_score=bm25_score,
        dense_score=dense_score,
        fused_score=0.5,
    )


def test_no_results_abstains():
    assessment = assess([])
    assert not assessment.sufficient
    assert assessment.reason == "no_results"


def test_low_dense_score_abstains_even_with_lexical_overlap():
    assessment = assess([_chunk(dense_score=DENSE_SCORE_FLOOR - 0.1, bm25_score=5.0)])
    assert not assessment.sufficient
    assert "below_floor" in assessment.reason


def test_strong_dense_match_without_keyword_overlap_is_sufficient():
    # Regression: this used to be refused ("no_lexical_overlap_with_top_result")
    assessment = assess([_chunk(dense_score=0.89, bm25_score=0.0)], query_language="en")
    assert assessment.sufficient


def test_uses_best_dense_score_not_first_result():
    # A keyword-only chunk can rank first in the fused order; the gate must
    # still see the strong dense match behind it.
    results = [_chunk(0.62, 12.0, "bm25_only"), _chunk(0.88, 0.0, "dense_match")]
    assert assess(results).sufficient


def test_non_english_query_language_does_not_change_the_decision():
    assert assess([_chunk(0.9, 0.0)], query_language="ta").sufficient
    assert not assess([_chunk(0.1, 0.0)], query_language="ta").sufficient
