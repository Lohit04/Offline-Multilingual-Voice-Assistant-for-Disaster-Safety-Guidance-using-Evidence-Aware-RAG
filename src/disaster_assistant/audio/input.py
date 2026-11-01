"""Microphone capture and WAV I/O.

Recording from a live microphone can only be verified interactively on the
user's own machine (not from this automated shell) -- record_from_mic has
not been exercised end-to-end yet. load_wav_file (used by the round-trip
test and by any pre-recorded-audio input path) has been.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf

from disaster_assistant import config


def record_from_mic(duration_s: float, output_path: str) -> None:
    audio = sd.rec(
        int(duration_s * config.AUDIO_SAMPLE_RATE),
        samplerate=config.AUDIO_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(output_path, audio, config.AUDIO_SAMPLE_RATE)


def load_wav_file(path: str) -> np.ndarray:
    audio, _sr = sf.read(path, dtype="float32")
    return audio
