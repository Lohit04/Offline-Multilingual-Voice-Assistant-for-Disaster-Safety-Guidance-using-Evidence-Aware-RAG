"""Offline text-to-speech, routed by language via config.TTS_CONFIG.

- Piper, for languages it ships a voice for (English/Hindi/Telugu/Malayalam).
- facebook/mms-tts-<code> (VITS, via transformers), for languages Piper has
  no voice for at all (Tamil, Kannada -- checked directly, 2026-10-10).
Every configured language has been verified to produce valid, non-silent
audio; none has been judged for quality/naturalness by a speaker of the
language. A language not in config.TTS_CONFIG falls back to
DEFAULT_TTS_LANGUAGE's voice rather than failing silently.
"""

import wave

import numpy as np
import soundfile as sf
from piper import PiperVoice

from disaster_assistant import config

_piper_voices: dict[str, PiperVoice] = {}
_mms_models: dict[str, tuple] = {}


def _get_piper_voice(voice_name: str) -> PiperVoice:
    if voice_name not in _piper_voices:
        model_path = config.PIPER_VOICE_DIR / f"{voice_name}.onnx"
        _piper_voices[voice_name] = PiperVoice.load(str(model_path))
    return _piper_voices[voice_name]


def _get_mms_model(model_name: str):
    if model_name not in _mms_models:
        from transformers import VitsModel, AutoTokenizer

        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _mms_models[model_name] = (model, tokenizer)
    return _mms_models[model_name]


def _synthesize_piper(text: str, output_path: str, voice_name: str) -> None:
    voice = _get_piper_voice(voice_name)
    with wave.open(output_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)


def _synthesize_mms(text: str, output_path: str, model_name: str) -> None:
    import torch

    # torch defaults to 8 threads on this 16-thread CPU; measured 2026-10-13
    # for a Tamil answer: 8 threads 4.5 s, 12 threads 3.8 s, 16 threads 3.5 s.
    # 12 leaves headroom for Ollama's CPU work running at the same time.
    torch.set_num_threads(12)
    model, tokenizer = _get_mms_model(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform
    audio = waveform.squeeze().numpy().astype(np.float32)
    sf.write(output_path, audio, model.config.sampling_rate)


def synthesize_to_wav(text: str, output_path: str, language: str = config.DEFAULT_TTS_LANGUAGE) -> None:
    voice_config = config.TTS_CONFIG.get(language, config.TTS_CONFIG[config.DEFAULT_TTS_LANGUAGE])
    if voice_config["engine"] == "mms":
        _synthesize_mms(text, output_path, voice_config["model"])
    else:
        _synthesize_piper(text, output_path, voice_config["voice"])
