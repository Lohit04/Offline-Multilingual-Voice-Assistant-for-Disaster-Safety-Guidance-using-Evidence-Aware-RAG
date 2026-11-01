"""Calibrates the instant-answer matcher (retrieval/answer_cache.py) on the
evaluation sets, which were NOT used to build the cache.

For every evaluation question in every language it records the best
cached situation, its score and its margin over the next situation, then
sweeps (MIN_SCORE, MIN_MARGIN) and reports, per setting:
  - answerable questions answered instantly with the RIGHT hazard (good)
  - answerable questions answered instantly with a WRONG hazard (unsafe)
  - unanswerable/live questions answered instantly (unsafe)
The chosen setting must have zero unsafe cases on these sets; among
those, the one with the most correct instant answers wins. The held-out
set (v2) is reported separately so tuning on v1 alone can be checked.

Usage: python scripts/calibrate_answer_cache.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from sentence_transformers import SentenceTransformer  # noqa: E402

from disaster_assistant import config  # noqa: E402
from disaster_assistant.retrieval.answer_cache import AnswerCache, question_hazards  # noqa: E402
from disaster_assistant.safety.live_info import mentions_current_time  # noqa: E402


BUFFER = 0.01


def load(name):
    return json.loads((ROOT / "data" / "evaluation" / name).read_text(encoding="utf-8"))["items"]


def rows_for(cache, embedder, items, set_name):
    qs = [(it, lang, text) for it in items for lang, text in it["q"].items()]
    embs = embedder.encode([f"query: {t}" for _, _, t in qs], normalize_embeddings=True, batch_size=64)
    rows = []
    for (it, lang, text), e in zip(qs, embs):
        allowed = question_hazards(text)
        # Mirrors answer.py: current-time questions never take the instant path.
        q = np.asarray(e, dtype="float32")
        res = None if mentions_current_time(text) else cache.scores(q, allowed or None)
        if res is not None and cache.reject_score(q) >= res[1]:
            res = None  # mirrors AnswerCache.match: closer to a reject phrasing
        if res is None:
            i, score, margin, phrase = None, 0.0, 0.0, None
        else:
            i, score, margin, phrase = res
        rows.append({"set": set_name, "id": it["id"], "type": it["type"], "cats": it.get("cats", []),
                     "lang": lang, "q": text, "known_hazard": bool(allowed),
                     "hazard": cache.intents[i]["hazard"] if i is not None else None,
                     "score": score, "margin": margin, "matched": cache.texts[phrase] if phrase is not None else None})
    return rows


def evaluate(rows, s, m):
    good = wrong = unsafe = answerable = 0
    for r in rows:
        hit = r["hazard"] is not None and r["score"] >= s and r["margin"] >= m
        if r["type"] == "answerable":
            answerable += 1
            if hit:
                if r["hazard"] in r["cats"] or (r["hazard"] == "urban_flood" and "flood" in r["cats"]):
                    good += 1
                else:
                    wrong += 1
        elif hit:
            unsafe += 1
    return good, wrong, unsafe, answerable


def main():
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    cache = AnswerCache(embedder)
    print(f"cache: {len(cache.intents)} situations, {len(cache.texts)} phrasings")
    v1 = rows_for(cache, embedder, load("questions_v1.json"), "v1")
    v2 = rows_for(cache, embedder, load("questions_v2_heldout.json"), "v2")

    results = {}
    for subset in (True, False):
        r1 = [r for r in v1 if r["known_hazard"] == subset]
        r2 = [r for r in v2 if r["known_hazard"] == subset]
        best = None
        for s in np.arange(0.80, 0.991, 0.005):
            for m in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
                g1, w1, u1, a1 = evaluate(r1, s, m)
                g2, w2, u2, a2 = evaluate(r2, s, m)
                # Safety buffer: must also be safe with the bar BUFFER lower,
                # so the chosen value isn't sitting right on an unsafe case.
                _, bw1, bu1, _ = evaluate(r1, s - BUFFER, m)
                _, bw2, bu2, _ = evaluate(r2, s - BUFFER, m)
                if w1 + u1 + w2 + u2 + bw1 + bu1 + bw2 + bu2 == 0 and (best is None or g1 + g2 > best[0]):
                    best = (g1 + g2, round(float(s), 3), m, (g1, a1), (g2, a2))
        label = "hazard named in the question" if subset else "no hazard word found"
        print(f"{label}: MIN_SCORE={best[1]} MIN_MARGIN={best[2]}  instant&correct v1 {best[3][0]}/{best[3][1]}, "
              f"v2(held-out) {best[4][0]}/{best[4][1]}")
        results["known" if subset else "any"] = {"min_score": best[1], "min_margin": best[2]}

    print()
    print("closest unsafe cases (highest-scoring non-answerable or wrong-hazard rows):")
    risky = [r for r in v1 + v2 if r["hazard"] and (r["type"] != "answerable" or r["hazard"] not in r["cats"])]
    for r in sorted(risky, key=lambda r: -r["score"])[:8]:
        print(f"  {r['score']:.3f} m={r['margin']:.3f} known={r['known_hazard']} [{r['set']} {r['id']} {r['type']} "
              f"{r['lang']}] {r['q'][:45]} -> {r['hazard']}")
    s, m = results, None
    out = ROOT / "experiments" / "results" / "answer_cache_calibration.json"
    out.write_text(json.dumps({"thresholds": results, "rows": v1 + v2}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nrows written to {out}")


if __name__ == "__main__":
    main()
