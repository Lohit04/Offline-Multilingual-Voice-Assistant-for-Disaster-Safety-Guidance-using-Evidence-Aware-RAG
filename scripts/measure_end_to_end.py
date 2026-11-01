"""End-to-end latency of the RUNNING web server, as a user experiences it.

Text: POST /api/ask for every evaluation question in every language; reports
how many got an instant answer and the time distribution for instant vs live.
Voice: TTS-synthesised clips of a few questions per language, POST
/api/voice_ask with the language chosen (the fast path) and with "auto".
Times are wall-clock per request, measured by this client.

Usage: python scripts/measure_end_to_end.py [--base https://127.0.0.1:8443]
Writes experiments/results/end_to_end_latency.json
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import requests
import urllib3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
urllib3.disable_warnings()


def summarise(label, xs):
    if not xs:
        return f"{label}: none"
    xs = sorted(xs)
    return (f"{label}: n={len(xs)} median {statistics.median(xs):.2f} s, "
            f"90th pct {xs[int(0.9 * (len(xs) - 1))]:.2f} s, max {xs[-1]:.2f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://127.0.0.1:8443")
    ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()
    items = []
    for name in ["questions_v1.json", "questions_v2_heldout.json"]:
        items += json.loads((ROOT / "data/evaluation" / name).read_text(encoding="utf-8"))["items"]
    out = {"text": [], "voice": []}

    if not args.skip_text:
        for it in items:
            for lang, q in it["q"].items():
                t = time.perf_counter()
                d = requests.post(f"{args.base}/api/ask", data={"question": q, "language": "auto"},
                                  verify=False, timeout=300).json()
                secs = time.perf_counter() - t
                out["text"].append({"id": it["id"], "type": it["type"], "lang": lang, "secs": round(secs, 3),
                                    "cached": d.get("cached", False), "abstained": d.get("abstained")})
        inst = [r["secs"] for r in out["text"] if r["cached"]]
        live = [r["secs"] for r in out["text"] if not r["cached"]]
        ans = [r for r in out["text"] if r["type"] == "answerable"]
        print(f"TEXT: {len(inst)}/{len(out['text'])} questions instant "
              f"({sum(r['cached'] for r in ans)}/{len(ans)} of answerable ones)")
        print("  " + summarise("instant", inst))
        print("  " + summarise("live AI", live))

    from disaster_assistant.audio.tts import synthesize_to_wav
    tmp = Path(tempfile.mkdtemp(prefix="e2e_"))
    by_id = {i["id"]: i for i in items}
    for qid in ["A01", "A02", "A03"]:
        for lang, q in by_id[qid]["q"].items():
            wav = tmp / f"{qid}_{lang}.wav"
            synthesize_to_wav(q, str(wav), lang)
            for mode in (lang, "auto"):
                t = time.perf_counter()
                with open(wav, "rb") as f:
                    d = requests.post(f"{args.base}/api/voice_ask", files={"audio": ("q.wav", f, "audio/wav")},
                                      data={"with_audio": "false", "language": mode}, verify=False, timeout=300).json()
                secs = time.perf_counter() - t
                out["voice"].append({"qid": qid, "lang": lang, "mode": "chosen" if mode != "auto" else "auto",
                                     "secs": round(secs, 3), "cached": d.get("cached", False),
                                     "transcript": d.get("transcript"), "error": d.get("error")})
    for mode in ("chosen", "auto"):
        rows = [r for r in out["voice"] if r["mode"] == mode]
        print(f"VOICE, language {mode}: {sum(r['cached'] for r in rows)}/{len(rows)} instant")
        print("  " + summarise("all", [r["secs"] for r in rows]))
        for lang in sorted({r["lang"] for r in rows}):
            lr = [r for r in rows if r["lang"] == lang]
            print(f"    {lang}: " + ", ".join(f"{r['secs']:.2f}s{'*' if r['cached'] else ''}" for r in lr))
    print("  (* = instant answer)")
    (ROOT / "experiments/results/end_to_end_latency.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
