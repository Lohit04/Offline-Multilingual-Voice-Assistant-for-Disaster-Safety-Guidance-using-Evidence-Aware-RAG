"""Offline speech-to-text via faster-whisper, GPU-accelerated when possible.

Measured directly (2026-10-10): CPU "small" took ~8-9s for an 8s clip;
CUDA "small" took ~1.6s for the same clip -- a real, large win, not a
marginal one. Getting CUDA working on Windows needed the
nvidia-cublas-cu12/nvidia-cudnn-cu12/nvidia-cuda-runtime-cu12 pip packages
(the driver alone isn't enough -- ctranslate2 needs the cuBLAS/cuDNN
runtime DLLs) *and* their bin/ directories added to the process PATH
specifically -- `os.add_dll_directory` alone was tried first and did NOT
work (ctranslate2's lazy CUDA library loading apparently doesn't go
through that mechanism), only prepending to PATH did.
"""

import os
import sysconfig
from dataclasses import dataclass
from pathlib import Path

from disaster_assistant import config
from disaster_assistant.generation.translation import detect_language_from_script


def _register_cuda_dll_dirs() -> None:
    """No-op if the nvidia-*-cu12 packages aren't installed -- CUDA just
    won't be available and get_model() falls back to CPU below."""
    site_packages = Path(sysconfig.get_paths()["purelib"])
    candidate_dirs = [
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
        site_packages / "nvidia" / "cuda_runtime" / "bin",
    ]
    existing = [str(d) for d in candidate_dirs if d.is_dir()]
    if existing:
        os.environ["PATH"] = os.pathsep.join(existing) + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402 -- must follow the PATH setup above

_model: WhisperModel | None = None


def _build_model(device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        download_root=str(config.WHISPER_MODEL_DIR),
    )


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = _build_model(config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
    return _model


# Short in-language prompts used only to RETRY when the first pass wrote the
# transcript in the wrong script (measured 2026-10-13: large-v3 wrote spoken
# Malayalam in Gurmukhi; adding a Malayalam prompt produced Malayalam script).
# They are domain vocabulary (flood / earthquake / cyclone), not instructions.
_SCRIPT_RETRY_PROMPTS = {
    "ml": "വെള്ളപ്പൊക്കം, ഭൂകമ്പം, ചുഴലിക്കാറ്റ് എന്നിവയെക്കുറിച്ചുള്ള ചോദ്യം.",
    "te": "వరద, భూకంపం, తుఫాను గురించి ప్రశ్న.",
    "kn": "ಪ್ರವಾಹ, ಭೂಕಂಪ, ಚಂಡಮಾರುತದ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ.",
    "ta": "வெள்ளம், நிலநடுக்கம், புயல் பற்றிய கேள்வி.",
    "hi": "बाढ़, भूकंप, चक्रवात के बारे में सवाल।",
}


@dataclass
class Transcription:
    text: str
    language: str
    language_probability: float
    retried_for_script: bool = False


def _run(model: WhisperModel, audio_path: str, **kwargs):
    # vad_filter drops non-speech: without it, silence was "transcribed" as
    # Whisper's stock hallucination ("Teksting av Nicolai Winther", detected as
    # Norwegian) in the 2026-10-13 stress test.
    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True, **kwargs)
    return " ".join(seg.text.strip() for seg in segments).strip(), info


_fast_model: WhisperModel | None = None


def _get_fast_model(device: str = config.WHISPER_DEVICE, compute_type: str = config.WHISPER_COMPUTE_TYPE):
    global _fast_model
    if _fast_model is None:
        _fast_model = WhisperModel(str(config.WHISPER_FAST_MODEL), device=device, compute_type=compute_type)
    return _fast_model


def _transcribe_known_language(audio_path: str, language: str) -> Transcription:
    """Fast path: turbo, told the language (see config.WHISPER_FAST_MODEL)."""
    global _fast_model
    try:
        text, _ = _run(_get_fast_model(), audio_path, language=language)
    except RuntimeError as e:
        if config.WHISPER_DEVICE == "cpu" or not any(k in str(e) for k in ("Library", "out of memory", "CUDA")):
            raise
        _fast_model = None
        text, _ = _run(_get_fast_model("cpu", "int8"), audio_path, language=language)
    return Transcription(text=text, language=language, language_probability=1.0)


def transcribe(audio_path: str, language: str | None = None) -> Transcription:
    """language=None: auto-detect with large-v3 (slower, needed because the
    fast model mislabels Telugu/Kannada/Malayalam). language given: the fast
    model, told the language."""
    if language and config.WHISPER_FAST_MODEL.exists():
        return _transcribe_known_language(audio_path, language)
    global _model
    model = get_model()
    try:
        text, info = _run(model, audio_path)
    except RuntimeError as e:
        # The CUDA library-loading failure this project hit during
        # development surfaces here (at compute time), not at model
        # construction -- fall back to CPU once, rather than crashing a
        # machine that doesn't have the nvidia-*-cu12 packages installed.
        # Also covers CUDA out-of-memory: large-v3 beside the LLM is tight on a
        # 6 GB GPU, and slow CPU transcription beats a failed request.
        msg = str(e)
        if config.WHISPER_DEVICE == "cpu" or not any(k in msg for k in ("Library", "out of memory", "CUDA")):
            raise
        _model = _build_model("cpu", "int8")
        model = _model
        text, info = _run(model, audio_path)

    retried = False
    lang = info.language
    if text and lang in _SCRIPT_RETRY_PROMPTS and detect_language_from_script(text) != lang:
        retry_text, _ = _run(model, audio_path, language=lang, initial_prompt=_SCRIPT_RETRY_PROMPTS[lang])
        if retry_text and detect_language_from_script(retry_text) == lang:
            text, retried = retry_text, True

    return Transcription(
        text=text,
        language=info.language,
        language_probability=info.language_probability,
        retried_for_script=retried,
    )
