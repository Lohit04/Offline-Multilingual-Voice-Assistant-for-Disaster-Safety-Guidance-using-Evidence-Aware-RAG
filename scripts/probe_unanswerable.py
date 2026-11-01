"""Retrieval-only scores for questions the knowledge base should NOT answer,
used to calibrate the abstention gate with more than the 4 unanswerable
topics in questions_v1.json. No LLM calls.

All questions are written by the AI assistant and unverified against the
corpus beyond a spot check -- if one of these turns out to be genuinely
covered by the NDMA documents, it is a mislabelled probe, not a gate failure.

Output: experiments/results/probe_unanswerable.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disaster_assistant.retrieval.hybrid import HybridRetriever

OFF_TOPIC = [
    "How do I invest in the stock market?",
    "Teach me Python programming",
    "How do I file my income tax return?",
    "What is the score of the cricket match?",
    "Give me a recipe for margherita pizza",
    "Which movie should I watch this weekend?",
    "How do I learn to play the guitar?",
    "Translate hello into French",
    "Which laptop is best for gaming?",
    "Who is the prime minister of Japan?",
]
NEAR_TOPIC_NOT_IN_KB = [
    "How do I treat a snake bite?",
    "What is the cure for dengue fever?",
    "What is the COVID vaccination schedule for children?",
    "How do I install solar panels on my roof?",
    "How do I pass a driving test?",
    "What is the history of the Taj Mahal?",
    "What is the population of Tamil Nadu?",
    "How does a nuclear power plant generate electricity?",
    "How do I learn to swim?",
    "How do I get a flood insurance claim approved?",
]


def main() -> None:
    r = HybridRetriever()
    out = []
    for kind, qs in [("off_topic", OFF_TOPIC), ("near_topic_not_in_kb", NEAR_TOPIC_NOT_IN_KB)]:
        for q in qs:
            res = r.search(q, top_k=5)
            out.append({
                "kind": kind, "question": q,
                "scores": [[x.dense_score, x.bm25_score] for x in res],
                "cats": [x.disaster_category for x in res],
                "top_text": res[0].text[:140] if res else "",
            })
            top = res[0]
            print(f"{kind[:4]} dense={top.dense_score:.3f} bm25={top.bm25_score:5.1f} maxdense5={max(x.dense_score for x in res):.3f} | {q}")
    (ROOT / "experiments" / "results" / "probe_unanswerable.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
