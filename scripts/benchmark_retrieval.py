"""Retrieval-only comparison of fusion strategies (no LLM calls).

Reuses the English search queries recorded by evaluate_pipeline.py (i.e. the
translator's output), so each variant sees exactly the queries production
sees. Proxy label: an answerable item "hits" if any top-k chunk's hazard
category is in the item's expected set (see data/evaluation/questions_v1.json
provenance note -- a coarse proxy, not human relevance judgements).

Variants:
  minmax   current production: 0.5*minmax(bm25)+0.5*minmax(dense) over a 30-candidate pool
  rrf      reciprocal rank fusion, k=60 (Cormack et al. 2009), no score normalisation
  dense    dense only
  bm25     BM25 only

Also reports, per variant, the "gate-score" problem: how often results[0]
is a BM25-only chunk whose dense_score was filled with 0 (the abstention gate
reads results[0].dense_score).

Usage: python scripts/benchmark_retrieval.py experiments/results/eval_v1_run1.jsonl
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disaster_assistant.retrieval.hybrid import HybridRetriever


def rrf(rank_lists, k=60):
    scores = {}
    for ranked in rank_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def run_variant(r: HybridRetriever, q: str, variant: str, k: int = 5):
    """Returns (ranked chunk ids, dense dict) for the variant."""
    bm = r._bm25_search(q, 30)
    de = r._dense_search(q, 30)
    if variant == "minmax":
        res = r.search(q, top_k=k)
        return [x.chunk_id for x in res], de
    if variant == "rrf":
        bm_rank = sorted(bm, key=bm.get, reverse=True)
        de_rank = sorted(de, key=de.get, reverse=True)
        return rrf([bm_rank, de_rank])[:k], de
    if variant == "dense":
        return sorted(de, key=de.get, reverse=True)[:k], de
    if variant == "bm25":
        return sorted(bm, key=bm.get, reverse=True)[:k], de
    raise ValueError(variant)


def main() -> None:
    rows = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [x for x in rows if not x.get("error") and x["type"] == "answerable"]
    r = HybridRetriever()
    # Several languages translate to the identical English string; dedupe work.
    cache = {}
    print(f"answerable (item,lang) rows: {len(rows)}\n")
    print(f"{'variant':8} {'cat-hit@5':>12} {'MRR(cat)':>10} {'top1-cat':>10} {'top1 dense=0':>14} {'max-dense<0.80':>16}")
    for variant in ["minmax", "rrf", "dense", "bm25"]:
        hit = top1 = zero_top = low_max = 0
        mrr = 0.0
        for row in rows:
            key = (variant, row["search_question"])
            if key not in cache:
                cache[key] = run_variant(r, row["search_question"], variant)
            ids, de = cache[key]
            cats = [r.chunks_by_id[i]["disaster_category"] for i in ids]
            want = set(row["expected_cats"])
            first = next((rank for rank, c in enumerate(cats, 1) if c in want), None)
            hit += first is not None
            top1 += first == 1
            mrr += 1 / first if first else 0
            zero_top += bool(ids) and de.get(ids[0], 0.0) == 0.0
            low_max += max((de.get(i, 0.0) for i in ids), default=0.0) < 0.80
        n = len(rows)
        print(f"{variant:8} {100*hit/n:11.1f}% {mrr/n:10.3f} {100*top1/n:9.1f}% {100*zero_top/n:13.1f}% {100*low_max/n:15.1f}%")


if __name__ == "__main__":
    main()
