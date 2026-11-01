"""Shared ASR -> retrieval/generation -> TTS sequencing, used by both the CLI
scripts and the Streamlit app so this logic lives in exactly one place."""

import time
from dataclasses import dataclass, field

from disaster_assistant import config
from disaster_assistant.audio.asr import transcribe
from disaster_assistant.audio.tts import synthesize_to_wav
from disaster_assistant.retrieval.hybrid import HybridRetriever
from disaster_assistant.generation.answer import Answer, answer_question

LOW_CONFIDENCE_LANGUAGE_THRESHOLD = 0.5


class EmptyTranscriptionError(Exception):
    """Raised specifically for silence/unintelligible audio -- deliberately
    NOT a bare ValueError: a real bug (2026-10-10) showed that some library
    exceptions (e.g. av.error.InvalidDataError, for a corrupted audio file)
    are themselves ValueError subclasses, so catching ValueError to mean
    "empty transcription" upstream was accidentally also swallowing genuine
    decode failures and reporting them as if they were just silence."""


@dataclass
class VoiceResult:
    transcript: str
    detected_language: str
    language_probability: float
    answer: Answer
    answer_language: str
    warnings: list[str]
    # Per-stage wall-clock seconds, so "latency is high" can be answered with
    # "which stage" instead of a guess. "asr"/"answer"/"tts" and "total".
    timings: dict[str, float] = field(default_factory=dict)


def process_voice_question(
    retriever: HybridRetriever, audio_path: str, output_wav_path: str | None, language: str | None = None
) -> VoiceResult:
    """language: the spoken language if the user chose it (fast speech
    recognition); None to auto-detect (slower)."""
    t_start = time.time()
    warnings = []
    timings = {}

    t0 = time.time()
    stt = transcribe(audio_path, language=language)
    timings["asr"] = time.time() - t0

    if not stt.text.strip():
        raise EmptyTranscriptionError("Empty transcription (silence or unintelligible audio) -- nothing to answer.")

    if stt.language_probability < LOW_CONFIDENCE_LANGUAGE_THRESHOLD:
        warnings.append(f"low language-detection confidence ({stt.language_probability:.2f})")

    if stt.language not in config.TTS_CONFIG:
        # Previously this fell back to answering in English, which turned noise
        # or an unsupported language into a confident-looking answer.
        raise EmptyTranscriptionError(
            f"Could not recognise a supported language (heard '{stt.language}'). "
            "Please speak English, Hindi, Tamil, Telugu, Kannada or Malayalam."
        )
    answer_language = stt.language

    t0 = time.time()
    answer = answer_question(retriever, stt.text, language=answer_language)
    timings["answer"] = time.time() - t0

    # output_wav_path=None skips speech synthesis: the web UI shows the text
    # answer immediately and fetches audio separately (/api/tts), because
    # Tamil/Kannada synthesis alone took ~10 s of a ~25 s voice round trip.
    t0 = time.time()
    if output_wav_path:
        synthesize_to_wav(answer.text, output_wav_path, language=answer_language)
    timings["tts"] = time.time() - t0

    timings["total"] = time.time() - t_start

    return VoiceResult(
        transcript=stt.text,
        detected_language=stt.language,
        language_probability=stt.language_probability,
        answer=answer,
        answer_language=answer_language,
        warnings=warnings,
        timings=timings,
    )
