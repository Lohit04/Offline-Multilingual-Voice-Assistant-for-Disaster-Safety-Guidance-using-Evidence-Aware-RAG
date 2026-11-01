"""The abstention gate reads RetrievedChunk.dense_score of the top result.
BM25-only candidates used to get dense_score=0.0, so the gate refused
answerable questions (cyclone-preparation abstained in 5/6 languages,
2026-10-13). Every returned chunk must carry its real cosine."""

import pytest

from disaster_assistant.retrieval.hybrid import HybridRetriever


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()


def test_exact_cosine_matches_faiss_search_scores(retriever):
    q = "How should I prepare for a cyclone?"
    q_emb = retriever._embed_query(q)
    dense = retriever._dense_search(q, 10, q_emb)
    for chunk_id, score in dense.items():
        assert retriever._exact_cosine(chunk_id, q_emb) == pytest.approx(score, abs=1e-5)


def test_no_returned_chunk_has_placeholder_zero_dense_score(retriever):
    for q in ["How should I prepare for a cyclone?", "What should I do if floodwater enters my house?"]:
        for r in retriever.search(q, top_k=5):
            assert r.dense_score > 0.5, (q, r.chunk_id, r.dense_score)
