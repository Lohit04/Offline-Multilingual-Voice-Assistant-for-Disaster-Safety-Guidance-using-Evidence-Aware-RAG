"""Hybrid BM25 + dense retrieval over the local chunk index.

Baseline fusion: normalize each ranker's scores to [0, 1] independently
(min-max over the candidates each ranker actually returned) and average them.
This is a simple, explainable starting point, not a validated weighting
scheme -- CLAUDE.md explicitly warns against combining retrieval signals with
arbitrary weights without evaluation, so treat the 0.5/0.5 split as a
baseline to be tested against BM25-only and dense-only, not a final answer.
"""

import json
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from disaster_assistant import config
from disaster_assistant.knowledge.indexing import lexical_tokens


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    disaster_category: str
    page: int
    text: str
    bm25_score: float
    dense_score: float
    fused_score: float


class HybridRetriever:
    def __init__(self):
        self.chunks_by_id: dict[str, dict] = {}
        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                self.chunks_by_id[c["chunk_id"]] = c

        bm25_ids = []
        tokenized_corpus = []
        with open(config.BM25_CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                bm25_ids.append(row["chunk_id"])
                tokenized_corpus.append(row["tokens"])
        self.bm25_ids = bm25_ids
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        self.faiss_ids = json.loads(config.FAISS_IDS_PATH.read_text(encoding="utf-8"))
        self._faiss_pos = {cid: i for i, cid in enumerate(self.faiss_ids)}

        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def _bm25_search(self, query: str, top_k: int) -> dict[str, float]:
        scores = self.bm25.get_scores(lexical_tokens(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return {self.bm25_ids[i]: float(scores[i]) for i in top_idx if scores[i] > 0}

    def _embed_query(self, query: str) -> np.ndarray:
        q_emb = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)
        return np.asarray(q_emb, dtype="float32")

    def _dense_search(self, query: str, top_k: int, q_emb: np.ndarray | None = None) -> dict[str, float]:
        if q_emb is None:
            q_emb = self._embed_query(query)
        scores, idx = self.faiss_index.search(q_emb, top_k)
        return {self.faiss_ids[i]: float(s) for s, i in zip(scores[0], idx[0]) if i != -1}

    def _exact_cosine(self, chunk_id: str, q_emb: np.ndarray) -> float:
        """True cosine of the query with one chunk. The index is a flat
        inner-product index over unit vectors, so this equals what a
        full-corpus search would have returned for that chunk."""
        vec = self.faiss_index.reconstruct(self._faiss_pos[chunk_id])
        return float(np.dot(q_emb[0], vec))

    @staticmethod
    def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 30) -> list[RetrievedChunk]:
        bm25_raw = self._bm25_search(query, candidate_pool)
        q_emb = self._embed_query(query)
        dense_raw = self._dense_search(query, candidate_pool, q_emb)
        bm25_norm = self._min_max_normalize(bm25_raw)
        dense_norm = self._min_max_normalize(dense_raw)

        all_ids = set(bm25_norm) | set(dense_norm)
        results = []
        for chunk_id in all_ids:
            bm25_s = bm25_norm.get(chunk_id, 0.0)
            dense_s = dense_norm.get(chunk_id, 0.0)
            fused = 0.5 * bm25_s + 0.5 * dense_s
            c = self.chunks_by_id[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=c["document_id"],
                    disaster_category=c["disaster_category"],
                    page=c["page"],
                    text=c["text"],
                    bm25_score=bm25_raw.get(chunk_id, 0.0),
                    # A chunk found only by BM25 is absent from dense_raw; report
                    # its real cosine rather than 0.0, because the abstention
                    # gate reads this field -- a 0.0 here made the gate refuse
                    # answerable questions whenever a keyword-only chunk
                    # ranked first (observed 2026-10-13: cyclone-preparation
                    # question abstained in 5/6 languages). Fusion above still
                    # uses only the rankers' own candidate scores, so ranking
                    # is unchanged.
                    dense_score=dense_raw[chunk_id] if chunk_id in dense_raw else self._exact_cosine(chunk_id, q_emb),
                    fused_score=fused,
                )
            )
        results.sort(key=lambda r: r.fused_score, reverse=True)
        return results[:top_k]
