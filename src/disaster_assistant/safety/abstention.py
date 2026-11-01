"""Evidence-sufficiency check, used to decide whether to answer or abstain.

This is a crude v0 heuristic, not a validated confidence model -- see
docs/research_gap.md. Empirically (scripts/ingest_documents.py output +
manual retrieval tests, 2026-10-10), raw dense cosine similarity from the
multilingual-e5-small embeddings did NOT clearly separate "answers this
question" from "topically adjacent but unhelpful" passages: both scored in
roughly the same 0.83-0.90 band. Requiring nonzero BM25 lexical overlap in
addition to a dense-score floor is a partial mitigation, not a fix.

Cross-lingual queries (e.g. Tamil questions against this English-only
corpus) are a special case: literal lexical overlap between a Tamil query
and English passages is structurally impossible (different scripts
entirely), so requiring it would mean the system could never answer a
non-English question regardless of how good the transcription or dense
match is -- found as a real bug 2026-10-10. For those, only the dense-score
floor applies; this leans entirely on the embedding model's cross-lingual
alignment being trustworthy, which has NOT been separately validated.
"""

from dataclasses import dataclass

from disaster_assistant import config
from disaster_assistant.retrieval.hybrid import RetrievedChunk

DENSE_SCORE_FLOOR = 0.80
# 0.80 is a coarse "clearly unrelated" cut, not a decision boundary: probes
# in scripts/probe_unanswerable.py show off-topic questions scoring up to
# 0.855 while answerable ones score ~0.85-0.91, so similarity alone cannot
# separate them. The real decision is made by
# safety/evidence_check.py (see its docstring); this floor only saves an LLM
# call for questions nothing in the corpus resembles.


@dataclass
class EvidenceAssessment:
    sufficient: bool
    reason: str
    top_results: list[RetrievedChunk]


def assess(results: list[RetrievedChunk], query_language: str = config.CORPUS_LANGUAGE) -> EvidenceAssessment:
    """First, cheap gate. `query_language` is accepted for compatibility but
    no longer used: the old "top result must share keywords with the query"
    rule was removed (2026-10-13) because it wrongly refused answerable
    questions whose best evidence was a dense-only match (dense 0.88-0.895,
    zero keyword overlap), and it is superseded by the LLM evidence check.

    Uses the best dense score among the returned chunks, not results[0]:
    the fused top result is not necessarily the most semantically similar.
    """
    if not results:
        return EvidenceAssessment(False, "no_results", results)

    best_dense = max(r.dense_score for r in results)
    if best_dense < DENSE_SCORE_FLOOR:
        return EvidenceAssessment(
            False, f"best_dense_score_below_floor ({best_dense:.3f} < {DENSE_SCORE_FLOOR})", results
        )

    return EvidenceAssessment(True, "ok", results)
