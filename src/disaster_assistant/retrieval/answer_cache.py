"""Instant answers: match a question to a situation prepared in advance by
scripts/build_answer_cache.py and return its stored, cited answer with no
LLM call at all.

Why (2026-10-26): web answers took 5-8 s, nearly all of it LLM time
(retrieval is 0.03 s); the target is under 3 s. Matching is a cosine
comparison between the question's multilingual-e5 embedding (the same
model the retriever already loaded) and every stored phrasing in all six
languages, so a Tamil question can match without the ~2.5 s LLM
translation step.

A match is accepted only if it is both strong (score >= MIN_SCORE) and
clearly ahead of the best phrasing from any OTHER situation (margin >=
MIN_MARGIN). Both values are calibrated on the evaluation sets by
scripts/calibrate_answer_cache.py -- see that script for the numbers --
with the rule that off-topic and live-information questions must never
get an instant answer. Anything that does not match goes to the normal
live pipeline, so a miss costs only ~30 ms.
"""

import json
import re
import threading
from dataclasses import dataclass

import numpy as np

from disaster_assistant import config
from disaster_assistant.generation.translation import HAZARD_GLOSSARY, detect_language_from_script

# Two-step matching (added after the first calibration showed e5 rating
# "what should I do during an earthquake?" in Kannada 0.97 similar to the
# FLOOD phrasings -- the shared sentence pattern outweighs the hazard word):
# 1. if the question names a hazard (English keyword, or the deterministic
#    Indian-language glossary from translation.py), only that hazard's
#    situations are candidates and the matcher just picks before/during/after;
# 2. otherwise everything is a candidate but a much stricter bar applies.
# Each pair calibrated separately by scripts/calibrate_answer_cache.py.
MIN_SCORE_KNOWN_HAZARD = 0.895
MIN_MARGIN_KNOWN_HAZARD = 0.0
MIN_SCORE_ANY = 0.93
MIN_MARGIN_ANY = 0.02

# English hazard words (prefix-matched) -> cache categories. Same table as
# the phone's OnDeviceRetrieval.HAZARDS; keep in sync.
EN_HAZARDS = [
    ("flood", ["flood", "urban_flood"]), ("inundat", ["flood", "urban_flood"]), ("waterlog", ["urban_flood"]),
    ("cyclone", ["cyclone"]), ("hurricane", ["cyclone"]), ("earthquake", ["earthquake"]), ("tremor", ["earthquake"]),
    ("landslide", ["landslide"]), ("mudslide", ["landslide"]), ("tsunami", ["tsunami"]), ("heat", ["heat_wave"]),
    ("cold", ["cold_wave"]), ("forest", ["forest_fire"]), ("wildfire", ["forest_fire"]), ("fire", ["fire", "forest_fire"]),
    ("lightning", ["lightning"]), ("thunder", ["thunderstorm_dust_storm_squall", "lightning"]),
    ("dust", ["thunderstorm_dust_storm_squall"]), ("avalanche", ["avalanche"]), ("drought", ["drought"]),
    ("chemical", ["chemical_emergency"]), ("nuclear", ["nuclear_radiological_emergency"]),
    ("radiation", ["nuclear_radiological_emergency"]), ("smog", ["smog_air_pollution"]),
    ("pollution", ["smog_air_pollution"]), ("epidemic", ["biological_emergency"]), ("pandemic", ["biological_emergency"]),
]
GLOSSARY_TO_CATS = {
    "flood": ["flood", "urban_flood"], "cyclone": ["cyclone"], "earthquake": ["earthquake"], "tsunami": ["tsunami"],
    "landslide": ["landslide"], "lightning": ["lightning", "thunderstorm_dust_storm_squall"],
    "heat wave": ["heat_wave"], "fire": ["fire", "forest_fire"],
}


def question_hazards(question: str) -> set[str]:
    """Cache categories the question names, in any of the six languages."""
    lang = detect_language_from_script(question)
    out: set[str] = set()
    if lang == "en":
        for word in re.findall(r"[a-z]+", question.lower()):
            for prefix, cats in EN_HAZARDS:
                if word.startswith(prefix):
                    out.update(cats)
    else:
        for term, name in HAZARD_GLOSSARY.get(lang, {}).items():
            if term in question:
                out.update(GLOSSARY_TO_CATS.get(name, []))
    return out


@dataclass
class CacheMatch:
    intent: dict
    score: float
    margin: float
    matched_text: str


class AnswerCache:
    def __init__(self, embedder, path=None):
        path = path or config.ANSWER_CACHE_PATH
        data = json.loads(path.read_text(encoding="utf-8"))
        self.intents = data["intents"]
        self.texts, self.owner = [], []
        for i, it in enumerate(self.intents):
            for qs in it["questions"].values():
                for q in qs:
                    self.texts.append(q)
                    self.owner.append(i)
        self.owner = np.array(self.owner)
        self.emb = np.asarray(
            embedder.encode([f"query: {t}" for t in self.texts], normalize_embeddings=True, batch_size=64),
            dtype="float32",
        )

        self.hazard_of = np.array([self.intents[o]["hazard"] for o in self.owner])
        # Reject set (scripts/add_cache_rejects.py): hazard-mentioning questions
        # that are not "what do I do" questions -- "What causes earthquakes?",
        # insurance, history. If a question is at least as close to one of
        # these as to its best situation, it gets no instant answer.
        rejects = [q for qs in data.get("rejects", {}).get("questions", {}).values() for q in qs]
        self.rej_emb = np.asarray(
            embedder.encode([f"query: {t}" for t in rejects], normalize_embeddings=True, batch_size=64),
            dtype="float32",
        ) if rejects else np.zeros((0, self.emb.shape[1]), dtype="float32")

    def reject_score(self, q_emb: np.ndarray) -> float:
        return float((self.rej_emb @ q_emb.reshape(-1)).max()) if len(self.rej_emb) else -1.0

    def scores(self, q_emb: np.ndarray, allowed: set[str] | None = None):
        """(best intent index, best score, margin over the best other candidate
        intent, best phrasing index), restricted to `allowed` hazards if given;
        None if no candidate exists."""
        sims = self.emb @ q_emb.reshape(-1)
        if allowed:
            sims = np.where(np.isin(self.hazard_of, list(allowed)), sims, -np.inf)
            if not np.isfinite(sims).any():
                return None
        best_idx = int(np.argmax(sims))
        best_intent = int(self.owner[best_idx])
        others = sims[(self.owner != best_intent) & np.isfinite(sims)]
        runner_up = float(others.max()) if others.size else 0.0
        return best_intent, float(sims[best_idx]), float(sims[best_idx]) - runner_up, best_idx

    def match(self, question: str, q_emb: np.ndarray):
        allowed = question_hazards(question)
        res = self.scores(q_emb, allowed or None)
        if res is None:
            return None
        intent_i, score, margin, phrase_i = res
        if self.reject_score(q_emb) >= score:
            return None
        min_s, min_m = ((MIN_SCORE_KNOWN_HAZARD, MIN_MARGIN_KNOWN_HAZARD) if allowed
                        else (MIN_SCORE_ANY, MIN_MARGIN_ANY))
        if score >= min_s and margin >= min_m:
            return CacheMatch(self.intents[intent_i], score, margin, self.texts[phrase_i])
        return None


_cache = None
_lock = threading.Lock()


def get_answer_cache(embedder):
    """Lazy singleton; returns None if the cache file has not been built."""
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                if not config.ANSWER_CACHE_PATH.exists():
                    return None
                _cache = AnswerCache(embedder)
    return _cache
