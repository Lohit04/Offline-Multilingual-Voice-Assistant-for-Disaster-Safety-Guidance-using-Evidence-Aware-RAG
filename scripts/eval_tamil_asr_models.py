"""Which Whisper model gives usable spoken Tamil for the phone?

Transcribes the 30 answerable Tamil evaluation questions (TTS clips) with
each candidate on the laptop (Hugging Face transformers, greedy decoding,
language forced to Tamil), then scores each transcript with the PHONE's
instant-answer matcher (calibrate_phone_matcher.Matcher, identical to
InstantAnswers.java). Vanilla openai/whisper-base is included as a control:
on the phone it got 5/30, so a matching number here shows the laptop proxy
is faithful.

Usage: python scripts/eval_tamil_asr_models.py <clips dir with index.json> [model ...]
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402
import soundfile as sf  # noqa: E402
import numpy as np  # noqa: E402

import calibrate_phone_matcher as phone  # noqa: E402

PHONE_KNOWN, PHONE_ANY, PHONE_ANY_MARGIN = 0.32, 0.30, 0.10


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
    return audio


def main():
    clips = Path(sys.argv[1])
    models = sys.argv[2:] or ["openai/whisper-base", "tamilnlpSLIIT/whisper-tiny-ta-v3", "mahimairaja/whisper-base-tamil"]
    idx = json.loads((clips / "index.json").read_text(encoding="utf-8"))
    data = json.loads((ROOT / "data/cache/answer_cache.json").read_text(encoding="utf-8"))
    intents = data["intents"]
    matcher = phone.Matcher(intents, data["rejects"]["questions"])

    def phone_match(q):
        if phone.mentions_current_time(q):
            return None
        i, s, mg, known = matcher.scores(q)
        if i is None:
            return None
        ok = s >= PHONE_KNOWN if known else (s >= PHONE_ANY and mg >= PHONE_ANY_MARGIN)
        return intents[i]["hazard"] if ok else None

    for name in models:
        proc = WhisperProcessor.from_pretrained(name, cache_dir=str(ROOT / "models/hf_tamil"))
        model = WhisperForConditionalGeneration.from_pretrained(name, cache_dir=str(ROOT / "models/hf_tamil"))
        model.eval()
        # Some community fine-tunes ship an old generation config without the
        # language table; use OpenAI's config for the same model size.
        if not hasattr(model.generation_config, "lang_to_id"):
            from transformers import GenerationConfig
            size = {384: "tiny", 512: "base", 768: "small", 1024: "medium"}[model.config.d_model]
            model.generation_config = GenerationConfig.from_pretrained(f"openai/whisper-{size}")
        good = wrong = 0
        secs = []
        examples = []
        for it in idx:
            feats = proc(load_audio(str(clips / it["file"])), sampling_rate=16000, return_tensors="pt").input_features
            t = time.perf_counter()
            with torch.no_grad():
                ids = model.generate(feats, language="ta", task="transcribe", max_new_tokens=120)
            secs.append(time.perf_counter() - t)
            text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            haz = phone_match(text)
            if haz and (haz in it["cats"] or (haz == "urban_flood" and "flood" in it["cats"])):
                good += 1
            elif haz:
                wrong += 1
            if len(examples) < 4:
                examples.append(text[:50])
        print(f"{name:38} instant&right {good}/30  wrong {wrong}  laptop-CPU {sum(secs) / len(secs):.2f} s/clip")
        for e in examples:
            print("     ", e)


if __name__ == "__main__":
    main()
