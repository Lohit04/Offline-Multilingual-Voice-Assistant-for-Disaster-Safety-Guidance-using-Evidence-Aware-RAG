"""Speed vs accuracy of faster-whisper beam_size on spoken test clips.

Clips are synthesised from the evaluation questions, so the true text is known
and a character error rate (CER) can be computed. CAVEAT: synthetic, clean,
single-speaker speech -- it ranks settings, it does not predict real-world WER.
"""
import difflib, json, re, sys, tempfile, time, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from disaster_assistant.audio import asr
from disaster_assistant.audio.tts import synthesize_to_wav
from disaster_assistant import config
from faster_whisper import WhisperModel

def norm(t):
    t = unicodedata.normalize("NFC", t.lower())
    return re.sub(r"[\s\W_]+", "", t)

def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return 1 - difflib.SequenceMatcher(None, r, h).ratio() if r else 0.0

items = {i["id"]: i for i in json.loads((ROOT / "data/evaluation/questions_v1.json").read_text(encoding="utf-8"))["items"]}
tmp = Path(tempfile.mkdtemp(prefix="beam_"))
clips = []
for qid in ("A01", "A02"):
    for lang in ("en", "hi", "ta", "te", "kn", "ml"):
        w = tmp / f"{qid}_{lang}.wav"
        synthesize_to_wav(items[qid]["q"][lang], str(w), lang)
        clips.append((lang, items[qid]["q"][lang], w))

model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cuda", compute_type=config.WHISPER_COMPUTE_TYPE, download_root=str(config.WHISPER_MODEL_DIR))
model.transcribe(str(clips[0][2]), beam_size=1)  # warm up
for beam in (5, 3, 1):
    per = {}
    t_all = 0.0
    for lang, ref, w in clips:
        t0 = time.perf_counter()
        segs, info = model.transcribe(str(w), beam_size=beam, vad_filter=True)
        hyp = " ".join(s.text.strip() for s in segs)
        dt = time.perf_counter() - t0
        t_all += dt
        per.setdefault(lang, []).append((cer(ref, hyp), dt))
    print(f"beam={beam}: total {t_all:.1f}s over {len(clips)} clips; mean CER by language: " +
          ", ".join(f"{l}={sum(c for c,_ in v)/len(v):.2f}" for l, v in per.items()) +
          f"  | overall CER={sum(c for v in per.values() for c,_ in v)/len(clips):.3f}", flush=True)
