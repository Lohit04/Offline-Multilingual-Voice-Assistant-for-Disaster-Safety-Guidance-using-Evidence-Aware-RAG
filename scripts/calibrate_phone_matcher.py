"""Calibrates the PHONE's instant-answer matcher, simulating exactly what
android-app/.../InstantAnswers.java does (no embedding model on the phone):

- normalise: lowercase; keep letters, digits and combining marks (Indic
  vowel signs are marks, not letters -- dropping them would split words),
  everything else becomes a space; pad with spaces
- character 3-grams, TF-IDF weighted (IDF over all stored phrasings),
  cosine similarity against every stored phrasing in all 6 languages
- accept if best >= MIN_SCORE and best minus best-other-situation >= MIN_MARGIN
- English-only guards (same as the server): a question that names a hazard
  may only match that hazard; live-information questions never match

Character n-grams instead of words because Tamil/Telugu/Kannada/Malayalam
attach endings to words (வெள்ளம் / வெள்ளத்தின் / வெள்ளத்தில்), so whole-word
matching misses most paraphrases.

Usage: python scripts/calibrate_phone_matcher.py
"""

import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from disaster_assistant.safety.live_info import mentions_current_time  # noqa: E402
from disaster_assistant.retrieval.answer_cache import question_hazards  # noqa: E402

N = 3
BUFFER = 0.05
# A question must beat its closest reject phrasing by this much (phone only:
# character n-grams separate topics less sharply than embeddings; chosen by
# probe_cache_rejects.py -- 4 off-topic leaks at 0.0, 1 at 0.05, no loss of
# safety controls).
REJECT_MARGIN = 0.05

# Must stay identical to OnDeviceRetrieval.HAZARDS (Java).
HAZARDS = [("flood", ["flood", "urban_flood"]), ("inundat", ["flood", "urban_flood"]), ("cyclone", ["cyclone"]),
           ("hurricane", ["cyclone"]), ("earthquake", ["earthquake"]), ("tremor", ["earthquake"]),
           ("landslide", ["landslide"]), ("mudslide", ["landslide"]), ("tsunami", ["tsunami"]),
           ("heat", ["heat_wave"]), ("cold", ["cold_wave"]), ("forest", ["forest_fire"]),
           ("wildfire", ["forest_fire"]), ("fire", ["fire", "forest_fire"]), ("lightning", ["lightning"]),
           ("thunder", ["thunderstorm_dust_storm_squall", "lightning"]), ("dust", ["thunderstorm_dust_storm_squall"]),
           ("avalanche", ["avalanche"]), ("drought", ["drought"]), ("chemical", ["chemical_emergency"]),
           ("nuclear", ["nuclear_radiological_emergency"]), ("radiation", ["nuclear_radiological_emergency"]),
           ("smog", ["smog_air_pollution"]), ("pollution", ["smog_air_pollution"]),
           ("epidemic", ["biological_emergency"]), ("pandemic", ["biological_emergency"])]


def normalise(text):
    out = []
    for ch in text.lower():
        out.append(ch if unicodedata.category(ch)[0] in "LNM" else " ")
    return " " + re.sub(r"\s+", " ", "".join(out)).strip() + " "


def grams(text):
    t = normalise(text)
    return Counter(t[i:i + N] for i in range(len(t) - N + 1))


def english_hazards(q):
    words = re.findall(r"[a-z]+", q.lower())
    out = set()
    for w in words:
        for k, cats in HAZARDS:
            if w.startswith(k):
                out.update(cats)
    return out


def is_english(q):
    return not any(ord(c) > 0x024F for c in q)


class Matcher:
    def __init__(self, intents, rejects=None):
        rejects = rejects or {}
        self.owner, self.vecs = [], []
        df = Counter()
        raw = []
        for i, it in enumerate(intents):
            for qs in it["questions"].values():
                for q in qs:
                    g = grams(q)
                    raw.append(g)
                    self.owner.append(i)
                    df.update(g.keys())
        n = len(raw)
        self.idf = {g: math.log(1 + n / c) for g, c in df.items()}
        self.vecs = [self._weigh(g) for g in raw]
        self.intents = intents
        # Reject phrasings: weighted with the same IDF (not part of it), as in Java.
        self.rej = [self._weigh(grams(q)) for qs in rejects.values() for q in qs]

    def reject_score(self, q):
        qv = self._weigh(grams(q))
        return max((sum(w * v.get(k, 0.0) for k, w in qv.items()) for v in self.rej), default=-1.0)

    def _weigh(self, g):
        v = {k: c * self.idf.get(k, math.log(1 + len(self.owner))) for k, c in g.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {k: x / norm for k, x in v.items()}

    def scores(self, q):
        qv = self._weigh(grams(q))
        # Same hazard detection as the web matcher (English keywords or the
        # Indian-language glossary); the Java port carries the same tables.
        allowed = question_hazards(q)
        best_per_intent = {}
        for vec, owner in zip(self.vecs, self.owner):
            if allowed and self.intents[owner]["hazard"] not in allowed:
                continue
            s = sum(w * vec.get(k, 0.0) for k, w in qv.items())
            if s > best_per_intent.get(owner, -1):
                best_per_intent[owner] = s
        if not best_per_intent:
            return None, 0.0, 0.0, bool(allowed)
        ranked = sorted(best_per_intent.items(), key=lambda kv: -kv[1])
        top_i, top_s = ranked[0]
        margin = top_s - (ranked[1][1] if len(ranked) > 1 else 0.0)
        if self.reject_score(q) >= top_s - REJECT_MARGIN:
            return None, 0.0, 0.0, bool(allowed)
        return top_i, top_s, margin, bool(allowed)


def main():
    data = json.loads((ROOT / "data/cache/answer_cache.json").read_text(encoding="utf-8"))
    intents = data["intents"]
    m = Matcher(intents, data.get("rejects", {}).get("questions", {}))
    print(f"{len(intents)} situations, {len(m.owner)} phrasings, {len(m.idf)} distinct 3-grams")
    rows = []
    for name in ["questions_v1.json", "questions_v2_heldout.json"]:
        for it in json.loads((ROOT / "data/evaluation" / name).read_text(encoding="utf-8"))["items"]:
            for lang, q in it["q"].items():
                live = mentions_current_time(q)
                i, s, mg, known = m.scores(q) if not live else (None, 0.0, 0.0, False)
                rows.append({"set": name[:12], "id": it["id"], "type": it["type"], "cats": it.get("cats", []),
                             "lang": lang, "q": q, "hazard": intents[i]["hazard"] if i is not None else None,
                             "score": s, "margin": mg, "known": known})

    def evaluate(rs, s, mg):
        good = wrong = unsafe = ans = 0
        per_lang = Counter()
        for r in rs:
            hit = r["hazard"] is not None and r["score"] >= s and r["margin"] >= mg
            if r["type"] == "answerable":
                ans += 1
                if hit and (r["hazard"] in r["cats"] or (r["hazard"] == "urban_flood" and "flood" in r["cats"])):
                    good += 1
                    per_lang[r["lang"]] += 1
                elif hit:
                    wrong += 1
            elif hit:
                unsafe += 1
        return good, wrong, unsafe, ans, per_lang

    total_good = Counter()
    for known in (True, False):
        rs = [r for r in rows if r["known"] == known]
        best = None
        for s in [x / 100 for x in range(10, 96, 2)]:
            for mg in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
                g, w, u, a, pl = evaluate(rs, s, mg)
                _, bw, bu, _, _ = evaluate(rs, s - BUFFER, mg)  # safety buffer, as in the web script
                if w + u + bw + bu == 0 and (best is None or g > best[0]):
                    best = (g, s, mg, a, pl)
        g, s, mg, a, pl = best
        total_good.update(pl)
        label = "hazard named" if known else "no hazard word"
        print(f"{label}: MIN_SCORE={s} MIN_MARGIN={mg}  instant&correct {g}/{a} answerable in this group")
    total = Counter(r["lang"] for r in rows if r["type"] == "answerable")
    print("instant & correct per language:", {l: f"{total_good[l]}/{total[l]}" for l in total})
    print("closest unsafe rows:")
    risky = [r for r in rows if r["hazard"] and (r["type"] != "answerable" or r["hazard"] not in r["cats"])]
    for r in sorted(risky, key=lambda r: -r["score"])[:6]:
        print(f"  {r['score']:.3f} m={r['margin']:.3f} [{r['id']} {r['type']} {r['lang']}] {r['q'][:45]} -> {r['hazard']}")


if __name__ == "__main__":
    main()
