"""Adds a REJECT set to the instant-answer cache: phrasings of questions that
name a hazard but are NOT "what do I do" safety questions (causes, history,
casualty counts, insurance, compensation, administration, property).

Why (2026-10-26): after calibration, probing showed "What causes
earthquakes?" and "Is it safe to invest in property in a flood zone?"
matching the earthquake/flood "during" answers on both matchers -- the
evaluation sets never asked such questions, so tuning could not see it.
A question whose best match is a reject phrasing gets no instant answer
and goes to the live pipeline instead.

The probe questions used to find the problem (scripts/probe_cache_rejects.py)
are deliberately NOT among these phrasings, so they still test whether the
reject set generalises. Translations are machine-generated, unreviewed.

Usage: python scripts/add_cache_rejects.py   (rewrites data/cache/answer_cache.json in place)
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from build_answer_cache import LANGS, translate_all  # noqa: E402

HAZARDS = ["a flood", "a cyclone", "an earthquake", "a tsunami", "a landslide", "a heat wave", "a fire", "lightning"]
TEMPLATES = [
    "Why does {h} happen?",
    "What is the scientific reason behind {h}?",
    "Tell me about the history of {h} in India.",
    "Which was the deadliest {h} ever recorded?",
    "How many people lost their lives in {h} last year?",
    "How can I claim insurance money for damage caused by {h}?",
    "Will the government pay compensation after {h}?",
    "Which ministry or agency is in charge of {h} management?",
    "How do scientists measure or predict {h}?",
    "Should I buy land in an area that gets {h}?",
    "Explain {h} for a school project.",
]


def main():
    path = ROOT / "data" / "cache" / "answer_cache.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    en = [t.format(h=h) for h in HAZARDS for t in TEMPLATES]
    pool = ThreadPoolExecutor(max_workers=4)
    # Batches of 11 (one hazard's templates) keep the numbered-list replies reliable.
    batches = [en[i:i + len(TEMPLATES)] for i in range(0, len(en), len(TEMPLATES))]
    rejects = {"en": en}
    for lang in LANGS[1:]:
        parts = list(pool.map(lambda b, l=lang: translate_all(b, l), batches))
        rejects[lang] = [q for part in parts for q in part]
        print(f"{lang}: {len(rejects[lang])} reject phrasings", flush=True)
    data["rejects"] = {
        "purpose": "Hazard-mentioning questions that are not safety-action questions; never answered instantly.",
        "questions": rejects,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"added {sum(len(v) for v in rejects.values())} reject phrasings")


if __name__ == "__main__":
    main()
