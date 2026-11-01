"""Summarises an evaluate_pipeline.py run and sweeps abstention thresholds
offline from the saved retrieval scores (no LLM calls).

Metrics (per language and overall):
  answerable : answered-rate (coverage), category-hit rate (a top-5 chunk's
               hazard category is in the item's expected set), wrong-script
               rate (answer not in the question's language).
  unanswerable: correct-abstention rate (higher is better).
  live       : abstained OR carried a disclaimer (any language) / disclaimer
               in the user's own language.
Threshold sweep: for a grid of dense-score floors (BM25 rule held as in
production), reports answer coverage on answerable items vs wrongful-answer
rate on unanswerable items -- a risk/coverage trade-off, not a calibrated
probability.

Usage: python scripts/summarize_eval.py experiments/results/eval_v1_run1.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from disaster_assistant.safety.live_info import LIVE_INFO_DISCLAIMER  # noqa: E402

LANGS = ["en", "hi", "ta", "te", "kn", "ml"]
# Any language's disclaimer counts as "has a disclaimer"; the first 25
# characters are enough to identify it. Disclaimers present in the CURRENT
# source are used, so run-1 rows (English fallback for hi/te/kn/ml) are
# judged against the new per-language text -- re-run to see the effect.
DISCLAIMER_MARKERS = [d[:25] for d in LIVE_INFO_DISCLAIMER.values()]
OWN_LANG_MARKERS = {lang: d[:25] for lang, d in LIVE_INFO_DISCLAIMER.items()}


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}% ({n}/{d})" if d else "  n/a"


def main() -> None:
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if not r.get("error")]
    print(f"rows={len(rows)} errors={len(rows) - len(ok)}  median latency={sorted(r['latency_s'] for r in ok)[len(ok)//2]}s\n")

    print("== per language ==")
    print(f"{'lang':4} {'ans-cov':>18} {'cat-hit|ans':>18} {'wrongscript':>14} {'unans-abst':>16} {'live-ok':>16} {'live-own-lang':>16} {'med-lat':>8}")
    for lang in LANGS:
        L = [r for r in ok if r["lang"] == lang]
        A = [r for r in L if r["type"] == "answerable"]
        answered = [r for r in A if not r["abstained"]]
        hit = [r for r in answered if set(r["retrieved_cats"]) & set(r["expected_cats"])]
        wrong_script = [r for r in answered if r["answer_script"] != lang]
        U = [r for r in L if r["type"] == "unanswerable"]
        Ua = [r for r in U if r["abstained"]]
        V = [r for r in L if r["type"] == "live"]
        v_ok = [r for r in V if r["abstained"] or any(m in r["answer"] for m in DISCLAIMER_MARKERS)]
        v_own = [r for r in V if r["abstained"] or OWN_LANG_MARKERS.get(lang, "@@none@@") in r["answer"]]
        lat = sorted(r["latency_s"] for r in L)
        print(f"{lang:4} {pct(len(answered), len(A)):>18} {pct(len(hit), len(answered)):>18} {pct(len(wrong_script), len(answered)):>14} "
              f"{pct(len(Ua), len(U)):>16} {pct(len(v_ok), len(V)):>16} {pct(len(v_own), len(V)):>16} {lat[len(lat)//2] if lat else 0:>7}s")

    print("\n== failures worth reading ==")
    for r in ok:
        if r["type"] == "answerable" and r["abstained"]:
            print(f"  WRONGFUL ABSTAIN {r['id']} {r['lang']} dense={r['top_dense']:.3f} bm25={r['top_bm25']:.2f} reason={r['abstention_reason']} | EN query: {r['search_question'][:70]}")
        if r["type"] == "unanswerable" and not r["abstained"]:
            print(f"  WRONGFUL ANSWER  {r['id']} {r['lang']} dense={r['top_dense']:.3f} bm25={r['top_bm25']:.2f} | EN query: {r['search_question'][:70]}")
        if r["type"] == "answerable" and not r["abstained"] and not (set(r["retrieved_cats"]) & set(r["expected_cats"])):
            print(f"  OFF-CATEGORY     {r['id']} {r['lang']} got={sorted(set(r['retrieved_cats']))} want={r['expected_cats']} | EN query: {r['search_question'][:70]}")
    for r in rows:
        if r.get("error"):
            print(f"  ERROR {r['id']} {r['lang']}: {r['error'][:100]}")

    print("\n== dense-floor sweep (BM25>0 rule as production, applied to the English search query) ==")
    A = [r for r in ok if r["type"] == "answerable"]
    U = [r for r in ok if r["type"] == "unanswerable"]
    print(f"{'floor':>6} {'answerable answered':>22} {'unanswerable answered (BAD)':>30}")
    for floor in [0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90]:
        def passes(r):
            return r["top_dense"] is not None and r["top_dense"] >= floor and (r["top_bm25"] or 0) > 0
        print(f"{floor:6.2f} {pct(sum(passes(r) for r in A), len(A)):>22} {pct(sum(passes(r) for r in U), len(U)):>30}")

    print("\n== score distributions ==")
    for label, grp in [("answerable", A), ("unanswerable", U)]:
        d = sorted(r["top_dense"] for r in grp if r["top_dense"] is not None)
        b = sorted(r["top_bm25"] for r in grp if r["top_bm25"] is not None)
        if d:
            print(f"  {label:13} dense min/med/max = {d[0]:.3f}/{d[len(d)//2]:.3f}/{d[-1]:.3f}   bm25 min/med/max = {b[0]:.2f}/{b[len(b)//2]:.2f}/{b[-1]:.2f}")


if __name__ == "__main__":
    main()
