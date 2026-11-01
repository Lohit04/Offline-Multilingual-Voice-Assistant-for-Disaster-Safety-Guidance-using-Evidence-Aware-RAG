"""Runs the text pipeline over data/evaluation/questions_v1.json in every
language and records, per (question, language): detected language, the
English search query the translator produced, retrieval scores, retrieved
hazard categories, whether it abstained, the answer, and latency.

Raw per-item scores are saved so abstention thresholds can be swept offline
(scripts/tune_threshold.py) without re-running the LLM.

Usage: python scripts/evaluate_pipeline.py [--langs en,hi] [--out PATH]
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disaster_assistant.retrieval.hybrid import HybridRetriever
from disaster_assistant.generation.answer import answer_question
from disaster_assistant.generation.translation import detect_language_from_script

DATASET = ROOT / "data" / "evaluation" / "questions_v1.json"
ALL_LANGS = ["en", "hi", "ta", "te", "kn", "ml"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(ALL_LANGS))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "eval_v1_run1.jsonl"))
    ap.add_argument("--dataset", default=str(DATASET), help="questions JSON (default: v1)")
    ap.add_argument("--no-evidence-check", action="store_true", help="ablation: disable the LLM evidence check")
    args = ap.parse_args()
    langs = args.langs.split(",")
    if args.no_evidence_check:
        from disaster_assistant import config
        config.EVIDENCE_CHECK_ENABLED = False

    items = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["items"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever()
    with out_path.open("w", encoding="utf-8") as out:
        for item in items:
            for lang in langs:
                question = item["q"][lang]
                t0 = time.perf_counter()
                try:
                    # language="auto" is what the Text tab sends.
                    ans = answer_question(retriever, question, language="auto")
                    err = None
                except Exception as e:  # record, don't crash the whole run
                    ans, err = None, f"{type(e).__name__}: {e}"
                elapsed = time.perf_counter() - t0

                rec = {
                    "id": item["id"], "type": item["type"], "lang": lang,
                    "expected_cats": item["cats"], "question": question,
                    "latency_s": round(elapsed, 2), "error": err,
                }
                if ans is not None:
                    top = ans.retrieved[0] if ans.retrieved else None
                    rec.update({
                        "answer_language": ans.answer_language,
                        "search_question": ans.search_question,
                        "abstained": ans.abstained,
                        "abstention_reason": ans.abstention_reason,
                        "top_dense": top.dense_score if top else None,
                        "top_bm25": top.bm25_score if top else None,
                        "retrieved_cats": [r.disaster_category for r in ans.retrieved],
                        "retrieved_docs": [r.document_id for r in ans.retrieved],
                        # per-result (dense, bm25) so abstention rules can be
                        # swept offline without re-running the LLM
                        "retrieved_scores": [[r.dense_score, r.bm25_score] for r in ans.retrieved],
                        "answer": ans.text,
                        "answer_script": detect_language_from_script(ans.text),
                        # instant (prepared) answer, and the hazards its sources cover
                        "cached": getattr(ans, "cached", False),
                        "source_cats": [s.disaster_category for s in ans.sources],
                    })
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                status = "ERR" if err else ("ABSTAIN" if ans.abstained else "ANSWER")
                print(f"{item['id']} {lang} {status} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
