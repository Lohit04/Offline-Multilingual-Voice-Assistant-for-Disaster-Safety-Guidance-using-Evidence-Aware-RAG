"""Probes both instant-answer matchers (web: retrieval/answer_cache.py; phone:
simulated InstantAnswers.java via calibrate_phone_matcher.py) with questions
that MENTION a hazard but are not "what do I do" questions. None of these
may get an instant answer. A few real safety questions are included as
controls; those should.

These probes are not used to build the reject set (add_cache_rejects.py),
so they test whether it generalises.

Usage: python scripts/probe_cache_rejects.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from sentence_transformers import SentenceTransformer  # noqa: E402

import calibrate_phone_matcher as phone  # noqa: E402
from disaster_assistant import config  # noqa: E402
from disaster_assistant.retrieval import answer_cache as ac  # noqa: E402
from disaster_assistant.safety.live_info import mentions_current_time  # noqa: E402

# Phone thresholds -- must equal InstantAnswers.java.
PHONE_KNOWN, PHONE_ANY, PHONE_ANY_MARGIN = 0.32, 0.30, 0.10

OFF_TOPIC = [
    "How do I file an insurance claim after a flood?", "Who is responsible for flood management in India?",
    "What is the history of cyclones in Odisha?", "How many people died in the 2004 tsunami?",
    "Can I get compensation for earthquake damage to my house?", "What causes earthquakes?",
    "Is it safe to invest in property in a flood zone?", "Why do cyclones form in the Bay of Bengal?",
    "How is a heat wave declared by IMD?", "Write a poem about a flood",
    "When did the Kedarnath flood happen?", "What is the Richter scale?",
    "How do floods affect farmers' income?", "What was the economic loss from cyclone Fani?",
    "बाढ़ बीमा का दावा कैसे करें?", "நிலநடுக்கம் ஏன் ஏற்படுகிறது?",
    "భూకంపాలు ఎందుకు వస్తాయి?", "சுனாமியில் எத்தனை பேர் இறந்தார்கள்?",
]
SAFETY = [
    "What should I do during a flood?", "How do I prepare for a cyclone?", "What do I do after an earthquake?",
    "How can I protect myself from a heat wave?", "வெள்ளத்தின் போது நான் என்ன செய்ய வேண்டும்?",
    "भूकंप आने पर क्या करें?",
]


def main():
    data = json.loads((ROOT / "data/cache/answer_cache.json").read_text(encoding="utf-8"))
    intents = data["intents"]
    pm = phone.Matcher(intents, data.get("rejects", {}).get("questions", {}))
    emb = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    web = ac.AnswerCache(emb)

    def web_hit(q):
        if mentions_current_time(q):
            return None
        m = web.match(q, np.asarray(emb.encode([f"query: {q}"], normalize_embeddings=True)[0], dtype="float32"))
        return m.intent["id"] if m else None

    def phone_hit(q):
        if mentions_current_time(q):
            return None
        i, s, mg, known = pm.scores(q)
        if i is None:
            return None
        ok = s >= PHONE_KNOWN if known else (s >= PHONE_ANY and mg >= PHONE_ANY_MARGIN)
        return intents[i]["id"] if ok else None

    bad = 0
    print("OFF-TOPIC (must NOT be instant):")
    for q in OFF_TOPIC:
        w, p = web_hit(q), phone_hit(q)
        bad += (w is not None) + (p is not None)
        print(f"  web={w or '-':28} phone={p or '-':28} {q}")
    good = 0
    print("SAFETY controls (should be instant):")
    for q in SAFETY:
        w, p = web_hit(q), phone_hit(q)
        good += (w is not None) + (p is not None)
        print(f"  web={w or '-':28} phone={p or '-':28} {q}")
    print(f"\nwrong instant answers on off-topic probes: {bad}; instant on safety controls: {good}/{2 * len(SAFETY)}")


if __name__ == "__main__":
    main()
