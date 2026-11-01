"""Builds the instant-answer cache: grounded, cited answers prepared in
advance for the situations the citizen Disaster Survival Guide covers
(hazard x before/during/after), in all 6 languages.

Why it exists (2026-10-26): measured web latency was 5-8 s per text
question (phone: 20-30 s) and almost all of it is LLM work -- retrieval
takes 0.03 s. The faculty target is under 3 s. Answers to the common
situations are therefore generated here, once, by the full server
pipeline (8B model + LLM evidence check), and at query time a question
that matches one of them is answered by lookup alone. Questions that
match nothing still go to the live pipeline.

Integrity rules:
- Evidence comes from the survival guide's chunks for that hazard only.
- A situation whose evidence fails the LLM evidence check is dropped,
  never answered from thin evidence.
- The evaluation sets (data/evaluation/) are NOT used here, so they stay
  an unbiased test of the matcher.
- Translations and answers are machine-generated and unreviewed; the
  output file says so.

Resumable: progress is written after every situation; rerunning skips
finished ones. Output: data/cache/answer_cache.json
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from disaster_assistant import config  # noqa: E402
from disaster_assistant.generation import ollama_client  # noqa: E402
from disaster_assistant.generation.prompts import LANGUAGE_NAMES, build_grounded_prompt  # noqa: E402
from disaster_assistant.retrieval.hybrid import HybridRetriever  # noqa: E402
from disaster_assistant.safety.evidence_check import evidence_supported  # noqa: E402

OUT = ROOT / "data" / "cache" / "answer_cache.json"
GUIDE = "ndma_disaster_survival_guide_2020"
LANGS = ["en", "hi", "ta", "te", "kn", "ml"]

# category -> (name with article, {phase: extra natural English phrasings})
HAZARDS = {
    "flood": ("a flood", {"during": ["Floodwater is entering my house, what should I do?",
                                     "Water is coming into my house, what should I do?"]}),
    "urban_flood": ("urban flooding", {"during": ["The streets in my city are waterlogged, what should I do?"]}),
    "cyclone": ("a cyclone", {"before": ["A cyclone warning has been issued, what should I do?"]}),
    "earthquake": ("an earthquake", {"during": ["The ground is shaking, what should I do?",
                                                "My house is shaking, what should I do?"]}),
    "tsunami": ("a tsunami", {}),
    "landslide": ("a landslide", {}),
    "heat_wave": ("a heat wave", {"during": ["It is extremely hot outside, how do I stay safe?",
                                             "How do I avoid heat stroke?"]}),
    "cold_wave": ("a cold wave", {}),
    "lightning": ("lightning", {"during": ["There is thunder and lightning outside, what should I do?"]}),
    "thunderstorm_dust_storm_squall": ("a thunderstorm or dust storm", {}),
    "fire": ("a fire", {"during": ["There is a fire in my house, what should I do?",
                                   "My building is on fire, what should I do?"]}),
    "forest_fire": ("a forest fire", {}),
    "drought": ("a drought", {}),
    "avalanche": ("an avalanche", {}),
    "chemical_emergency": ("a chemical leak from a factory", {}),
    "nuclear_radiological_emergency": ("a nuclear or radiation emergency", {}),
    "biological_emergency": ("an epidemic or disease outbreak", {}),
    "smog_air_pollution": ("severe smog or air pollution", {}),
}

TEMPLATES = {
    "before": ["How do I prepare for {h}?", "What should I do before {h}?",
               "How can I keep my family safe before {h}?"],
    "during": ["What should I do during {h}?", "What to do in {h}?", "How do I stay safe during {h}?"],
    "after": ["What should I do after {h}?", "What precautions should I take after {h}?"],
}

TRANSLATE_PROMPT = """Translate this English question into {lang}, the way a native {lang} speaker \
would naturally ask it. Output ONLY the {lang} question, nothing else.

English: {text}
{lang}:"""


def translate(text: str, lang: str) -> str:
    name = LANGUAGE_NAMES[lang]
    out = ollama_client.generate(TRANSLATE_PROMPT.format(lang=name, text=text), max_tokens=120)
    return out.strip().strip('"').splitlines()[0].strip()


BATCH_PROMPT = """Translate each numbered English question into {lang}, the way a native {lang} \
speaker would naturally ask it. Reply with ONLY the numbered {lang} questions, one per line, \
same numbering, nothing else.

{items}"""


def translate_all(texts: list[str], lang: str) -> list[str]:
    """One LLM call per language instead of one per phrasing (the first run
    took ~5 min per situation with per-phrasing calls). Falls back to
    one-by-one if the reply doesn't come back as the same numbered list."""
    name = LANGUAGE_NAMES[lang]
    items = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
    reply = ollama_client.generate(BATCH_PROMPT.format(lang=name, items=items), max_tokens=600)
    got = {}
    for line in reply.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
        if m:
            got[int(m.group(1))] = m.group(2).strip().strip('"')
    if sorted(got) == list(range(1, len(texts) + 1)):
        return [got[i] for i in range(1, len(texts) + 1)]
    return [translate(t, lang) for t in texts]


def guide_evidence(retriever: HybridRetriever, question: str, hazard: str, k: int = 5):
    hits = retriever.search(question, top_k=40, candidate_pool=80)
    hits.sort(key=lambda r: r.fused_score, reverse=True)
    own = [r for r in hits if r.document_id == GUIDE and r.disaster_category == hazard]
    return own[:k]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if OUT.exists():
        for it in json.loads(OUT.read_text(encoding="utf-8")).get("intents", []):
            done[it["id"]] = it
    dropped = []

    retriever = HybridRetriever()
    pool = ThreadPoolExecutor(max_workers=4)

    for hazard, (name, extras) in HAZARDS.items():
        for phase, templates in TEMPLATES.items():
            iid = f"{hazard}.{phase}"
            if iid in done:
                continue
            t0 = time.time()
            en = [t.format(h=name) for t in templates] + extras.get(phase, [])
            canonical = en[0]
            evidence = guide_evidence(retriever, canonical, hazard)
            if not evidence or not evidence_supported(canonical, evidence):
                dropped.append(iid)
                print(f"[drop] {iid}: survival guide evidence does not support it", flush=True)
                continue

            questions = {"en": en, **dict(zip(LANGS[1:], pool.map(lambda l: translate_all(en, l), LANGS[1:])))}

            answers = dict(zip(LANGS, pool.map(
                lambda l: ollama_client.generate(build_grounded_prompt(canonical, evidence, answer_language=l)).strip(),
                LANGS)))

            done[iid] = {
                "id": iid, "hazard": hazard, "phase": phase, "canonical": canonical,
                "questions": questions, "answers": answers,
                "sources": [{"document_id": r.document_id, "page": r.page,
                             "disaster_category": r.disaster_category, "text": r.text} for r in evidence],
            }
            _save(done)
            print(f"[ok]   {iid}  ({time.time() - t0:.0f}s)", flush=True)

    _save(done)
    print(f"\nDone: {len(done)} situations cached; dropped this run: {dropped}")


def _save(done: dict) -> None:
    OUT.write_text(json.dumps({
        "version": 1,
        "created": date.today().isoformat(),
        "model": config.OLLAMA_MODEL,
        "provenance": ("Answers generated by the project's own grounded pipeline from the NDMA Disaster "
                       "Survival Guide (2020), one hazard's passages per answer, each passing the LLM "
                       "evidence check. Non-English questions and answers are MACHINE TRANSLATIONS, not "
                       "reviewed by native speakers. Not reviewed by a disaster-management expert."),
        "intents": list(done.values()),
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
