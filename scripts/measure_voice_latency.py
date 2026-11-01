"""Per-stage voice latency on a RUNNING server, using TTS-synthesised clips.

Prints ASR / translate / retrieve / evidence-check / generate / TTS seconds per
clip, for a cold pass and a warm pass, so optimisation targets the stage that
actually dominates. Usage:
    python scripts/measure_voice_latency.py [--base http://localhost:8001] [--rounds 2]
Writes experiments/results/voice_latency.json
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASET = ROOT / "data" / "evaluation" / "questions_v1.json"
CLIPS = [("A02", "en"), ("A01", "ta"), ("A02", "hi"), ("A02", "te"), ("A02", "kn"), ("A02", "ml")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--langs", default="")
    args = ap.parse_args()

    from disaster_assistant.audio.tts import synthesize_to_wav

    items = {i["id"]: i for i in json.loads(DATASET.read_text(encoding="utf-8"))["items"]}
    tmp = Path(tempfile.mkdtemp(prefix="lat_audio_"))
    clips = []
    for qid, lang in CLIPS:
        if args.langs and lang not in args.langs.split(","):
            continue
        wav = tmp / f"{qid}_{lang}.wav"
        synthesize_to_wav(items[qid]["q"][lang], str(wav), lang)
        clips.append((qid, lang, wav))

    rows = []
    for rnd in range(1, args.rounds + 1):
        print(f"--- round {rnd} ---")
        print("(total = seconds until the TEXT answer is on screen; audio = extra seconds until speech is ready)")
        print(f"{'clip':8} {'total':>6} {'asr':>5} {'transl':>6} {'retr':>5} {'check':>6} {'gen':>5} {'audio':>5}  {'answer'}")
        for qid, lang, wav in clips:
            t0 = time.perf_counter()
            # Same flow as the web UI: text first (with_audio=false), audio fetched after.
            r = requests.post(f"{args.base}/api/voice_ask", data={"with_audio": "false"},
                              files={"audio": (wav.name, open(wav, "rb"), "audio/wav")}, timeout=300).json()
            wall = time.perf_counter() - t0
            audio_s = None
            if r.get("text") and r.get("answer_language"):
                t1 = time.perf_counter()
                requests.post(f"{args.base}/api/tts", data={"text": r["text"], "language": r["answer_language"]}, timeout=300)
                audio_s = time.perf_counter() - t1
            tm, at = r.get("timings", {}), r.get("answer_timings", {})
            row = {"round": rnd, "clip": f"{qid}_{lang}", "wall": round(wall, 1), "asr": tm.get("asr"), "answer": tm.get("answer"),
                   "tts": audio_s, **{k: at.get(k) for k in ("translate", "retrieve", "evidence_check", "generate")},
                   "abstained": r.get("abstained")}
            rows.append(row)
            f = lambda v: f"{v:5.1f}" if isinstance(v, (int, float)) else "   - "
            print(f"{row['clip']:8} {wall:6.1f} {f(row['asr'])} {f(row['translate'])}  {f(row['retrieve'])} {f(row['evidence_check'])}  {f(row['generate'])} {f(row['tts'])}  {'abstain' if row['abstained'] else 'answer'}")
    (ROOT / "experiments" / "results" / "voice_latency.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
