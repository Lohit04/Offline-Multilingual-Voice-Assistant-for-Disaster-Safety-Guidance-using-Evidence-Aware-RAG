"""Accuracy of the final system, from evaluate_pipeline.py runs.

Per item, "correct behaviour" means:
  answerable   -> answered (not abstained) AND grounded in the right hazard
                  (a source / retrieved passage's hazard category is in the
                  item's expected categories);
  unanswerable -> abstained (off-topic refused);
  live         -> abstained OR the answer carries the live-information
                  disclaimer (in any language).
Also reported: instant-answer precision (instant answers whose sources are
the right hazard), answer-language accuracy (answer written in the
question's language), and per-language breakdowns.

Caveat (as everywhere in this project): the questions were written by the
AI assistant and translations are not reviewed by native speakers; "right
hazard" is a proxy label, not a human judgement of the answer text.

Usage: python scripts/analyze_final_eval.py RUN.jsonl [RUN2.jsonl ...] [--out results.json]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from disaster_assistant.safety.live_info import LIVE_INFO_DISCLAIMER  # noqa: E402

MARKERS = [v[:25] for v in LIVE_INFO_DISCLAIMER.values()]
URBAN = {"flood": {"urban_flood"}, "urban_flood": {"flood"}}


def right_hazard(rec):
    exp = set(rec.get("expected_cats") or [])
    for c in list(exp):
        exp |= URBAN.get(c, set())
    got = set(rec.get("source_cats") or []) | set(rec.get("retrieved_cats") or [])
    if rec.get("cached"):
        got = set(rec.get("source_cats") or [])
    return bool(exp & got)


def behaviour_ok(rec):
    if rec.get("error"):
        return False
    t = rec["type"]
    if t == "answerable":
        return (not rec["abstained"]) and right_hazard(rec)
    if t == "unanswerable":
        return bool(rec["abstained"])
    if t == "live":
        return bool(rec["abstained"]) or any(m in (rec.get("answer") or "") for m in MARKERS)
    return False


def summarize(rows):
    out = {"n": len(rows)}
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    ok = [behaviour_ok(r) for r in rows]
    out["overall_correct"] = [sum(ok), len(ok)]
    ans = by_type["answerable"]
    answered = [r for r in ans if not r.get("abstained") and not r.get("error")]
    out["answerable_answered"] = [len(answered), len(ans)]
    out["answerable_right_hazard"] = [sum(right_hazard(r) for r in answered), len(answered)]
    out["answerable_correct"] = [sum(behaviour_ok(r) for r in ans), len(ans)]
    inst = [r for r in answered if r.get("cached")]
    out["instant_answers"] = [len(inst), len(ans)]
    out["instant_precision"] = [sum(right_hazard(r) for r in inst), len(inst)]
    out["answer_language_ok"] = [sum(r.get("answer_script") == r["lang"] for r in answered), len(answered)]
    off = by_type["unanswerable"]
    out["offtopic_refused"] = [sum(behaviour_ok(r) for r in off), len(off)]
    out["offtopic_instant"] = [sum(bool(r.get("cached")) for r in off), len(off)]
    live = by_type["live"]
    out["live_handled"] = [sum(behaviour_ok(r) for r in live), len(live)]
    out["live_instant"] = [sum(bool(r.get("cached")) for r in live), len(live)]
    lat = sorted(r["latency_s"] for r in rows)
    out["median_latency_s"] = lat[len(lat) // 2] if lat else None
    lat_i = sorted(r["latency_s"] for r in inst)
    out["median_latency_instant_s"] = lat_i[len(lat_i) // 2] if lat_i else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    result = {}
    for path in a.runs:
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        res = {"all": summarize(rows), "by_language": {}}
        for lang in sorted({r["lang"] for r in rows}):
            res["by_language"][lang] = summarize([r for r in rows if r["lang"] == lang])
        result[Path(path).stem] = res
        s = res["all"]
        pct = lambda p: f"{p[0]}/{p[1]} ({100 * p[0] / p[1]:.1f}%)" if p[1] else "-"
        print(f"\n== {Path(path).name}")
        for k in ("overall_correct", "answerable_answered", "answerable_right_hazard", "answerable_correct",
                  "instant_answers", "instant_precision", "answer_language_ok", "offtopic_refused",
                  "offtopic_instant", "live_handled", "live_instant"):
            print(f"  {k:26} {pct(s[k])}")
        print(f"  median latency {s['median_latency_s']} s, instant {s['median_latency_instant_s']} s")
        for lang, ls in res["by_language"].items():
            print(f"  [{lang}] correct {pct(ls['overall_correct'])}  answered {pct(ls['answerable_answered'])}  "
                  f"instant {pct(ls['instant_answers'])}  refused {pct(ls['offtopic_refused'])}  live {pct(ls['live_handled'])}")
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
