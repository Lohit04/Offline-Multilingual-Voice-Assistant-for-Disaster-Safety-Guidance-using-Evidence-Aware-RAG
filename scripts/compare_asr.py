"""Speed and accuracy of Whisper large-v3 vs large-v3-turbo on the same clips.

Why (2026-10-26): with instant answers, a voice question's latency is
almost entirely speech recognition (large-v3: ~3-6 s per short clip), and
the target is under 3 s. Turbo keeps large-v3's encoder but has a much
smaller decoder.

Clips are TTS-synthesised from evaluation questions (the project's own
Piper/MMS voices): clean, single speaker. That is good for a like-for-like
speed comparison and for catching wrong-script output, but says nothing
about real noisy human speech -- state that alongside any number from here.

Per model and language it reports: warm seconds per clip, whether the
output is in the right script, and character similarity to the reference.

Usage: python scripts/compare_asr.py [--models large-v3,models/whisper/large-v3-turbo-ct2]
Writes experiments/results/asr_comparison.json
"""

import argparse
import difflib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

# Importing asr first puts the CUDA runtime DLLs on PATH (see asr.py's docstring).
import disaster_assistant.audio.asr  # noqa: E402,F401
from faster_whisper import WhisperModel  # noqa: E402

from disaster_assistant import config  # noqa: E402
from disaster_assistant.audio.tts import synthesize_to_wav  # noqa: E402
from disaster_assistant.generation.translation import detect_language_from_script  # noqa: E402

QIDS = ["A01", "A02", "A03"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="large-v3,models/whisper/large-v3-turbo-ct2")
    args = ap.parse_args()

    items = {i["id"]: i for i in json.loads((ROOT / "data/evaluation/questions_v1.json").read_text(encoding="utf-8"))["items"]}
    tmp = Path(tempfile.mkdtemp(prefix="asr_cmp_"))
    clips = []
    for qid in QIDS:
        for lang, text in items[qid]["q"].items():
            if lang not in config.TTS_CONFIG:
                continue
            wav = tmp / f"{qid}_{lang}.wav"
            synthesize_to_wav(text, str(wav), lang)
            clips.append((qid, lang, text, str(wav)))
    print(f"{len(clips)} clips")

    report = {}
    for name in args.models.split(","):
        path = str(ROOT / name) if "/" in name else name
        model = WhisperModel(path, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE,
                             download_root=str(config.WHISPER_MODEL_DIR))
        list(model.transcribe(clips[0][3], beam_size=5, vad_filter=True)[0])  # warm-up
        rows = []
        for qid, lang, ref, wav in clips:
            t = time.perf_counter()
            segs, info = model.transcribe(wav, beam_size=5, vad_filter=True)
            text = " ".join(s.text for s in segs).strip()
            secs = time.perf_counter() - t
            rows.append({"qid": qid, "lang": lang, "secs": round(secs, 2), "text": text,
                         "detected": info.language,
                         "right_script": detect_language_from_script(text) == lang,
                         "similarity": round(difflib.SequenceMatcher(None, ref, text).ratio(), 2)})
        report[name] = rows
        del model
        print(f"\n== {name}")
        for lang in sorted({r["lang"] for r in rows}):
            rs = [r for r in rows if r["lang"] == lang]
            print(f"  {lang}: {sum(r['secs'] for r in rs) / len(rs):.2f} s/clip  right script "
                  f"{sum(r['right_script'] for r in rs)}/{len(rs)}  similarity "
                  f"{sum(r['similarity'] for r in rs) / len(rs):.2f}")
        print(f"  all: {sum(r['secs'] for r in rows) / len(rows):.2f} s/clip")
    out = ROOT / "experiments/results/asr_comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
