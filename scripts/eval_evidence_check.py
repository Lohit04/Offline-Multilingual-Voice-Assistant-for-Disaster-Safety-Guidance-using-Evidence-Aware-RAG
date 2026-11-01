"""Measures the LLM evidence check on labelled retrieval results (no answer
generation): probes (should be NO) vs answerable English queries from an
evaluation run (should be YES). Reports accuracy by dense-score band so the
grey-zone cut-offs can be chosen from data.

Usage: python scripts/eval_evidence_check.py experiments/results/eval_v1_run2.jsonl
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disaster_assistant.retrieval.hybrid import HybridRetriever
from disaster_assistant.safety.evidence_check import evidence_supported


def main() -> None:
    r = HybridRetriever()
    rows = []
    # Negatives: probes
    for p in json.loads((ROOT / "experiments/results/probe_unanswerable.json").read_text(encoding="utf-8")):
        rows.append(("NO", p["kind"], p["question"]))
    # Positives: distinct English search queries of answerable items from the run
    seen = set()
    for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
        x = json.loads(l)
        if x["type"] == "answerable" and x["lang"] == "en" and x["search_question"] not in seen:
            seen.add(x["search_question"])
            rows.append(("YES", "answerable_en", x["search_question"]))

    out = []
    for expected, kind, q in rows:
        res = r.search(q, top_k=5)
        t0 = time.perf_counter()
        got = evidence_supported(q, res)
        dt = time.perf_counter() - t0
        top_dense = max(x.dense_score for x in res)
        out.append({"expected": expected, "kind": kind, "q": q, "got": "YES" if got else "NO", "max_dense5": top_dense, "s": round(dt, 1)})
        flag = "" if (got == (expected == "YES")) else "   <-- WRONG"
        print(f"exp={expected:3} got={'YES' if got else 'NO ':3} maxdense={top_dense:.3f} {dt:4.1f}s {kind[:10]:10} | {q[:60]}{flag}", flush=True)

    tp = sum(o["expected"] == "YES" and o["got"] == "YES" for o in out)
    tn = sum(o["expected"] == "NO" and o["got"] == "NO" for o in out)
    P = sum(o["expected"] == "YES" for o in out)
    N = sum(o["expected"] == "NO" for o in out)
    print(f"\nYES recall (answerable kept): {tp}/{P}   NO specificity (off-topic refused): {tn}/{N}")
    (ROOT / "experiments/results/evidence_check_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
