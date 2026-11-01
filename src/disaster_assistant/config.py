from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_METADATA = ROOT / "data" / "metadata"
DOCUMENTS_REGISTRY = DATA_METADATA / "documents.json"
CHUNKS_PATH = DATA_PROCESSED / "chunks.jsonl"
# Instant answers prepared in advance (scripts/build_answer_cache.py); see
# retrieval/answer_cache.py. Set ANSWER_CACHE_ENABLED = False to force every
# question through the live pipeline (e.g. for ablations).
ANSWER_CACHE_PATH = ROOT / "data" / "cache" / "answer_cache.json"
ANSWER_CACHE_ENABLED = True

INDEX_VECTOR_DIR = ROOT / "indexes" / "vector"
INDEX_LEXICAL_DIR = ROOT / "indexes" / "lexical"
FAISS_INDEX_PATH = INDEX_VECTOR_DIR / "faiss.index"
FAISS_IDS_PATH = INDEX_VECTOR_DIR / "chunk_ids.json"
BM25_CORPUS_PATH = INDEX_LEXICAL_DIR / "tokenized_corpus.jsonl"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
CORPUS_LANGUAGE = "en"  # all ingested source documents are in English

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"
# Ollama's default keep_alive is 5 minutes, after which it unloads the model
# and the next call pays a ~30s reload cost (measured directly, 2026-10-10:
# 32s cold vs 0.5s warm). Keeping it loaded longer avoids that for repeated
# use in one sitting; it still frees the ~3GB VRAM after this long idle.
OLLAMA_KEEP_ALIVE = "30m"

# LLM yes/no check that the retrieved passages actually answer the question
# (safety/evidence_check.py). Switchable so experiments can ablate it.
EVIDENCE_CHECK_ENABLED = True
# Generate the answer while the evidence check runs (discard if the check says no).
SPECULATIVE_GENERATION = True

CHUNK_WORD_SIZE = 180
CHUNK_WORD_OVERLAP = 40

# "large-v3" chosen by measurement (2026-10-13, spoken test clips): "small" and
# "medium" transcribed Telugu/Kannada/Malayalam as Devanagari transliteration
# (correct script on 4/10 clips for medium); large-v3 produced correct Telugu
# and Kannada text (8/10 clips overall; Malayalam still unreliable). Costs
# ~3-6 s per short clip vs ~1.6 s for small, and ~2.5 GB VRAM next to the LLM.
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_MODEL_DIR = ROOT / "models" / "whisper"
# Fast path when the user has CHOSEN the spoken language (web/phone language
# chips): Whisper large-v3-turbo, told the language. Measured 2026-10-26
# (scripts/compare_asr.py + a forced-language run, synthetic clips, RTX 4050):
# 0.6-1.2 s per clip vs 1.3-3.5 s for large-v3, right script in all six
# languages when told the language (similarity: en 1.00, hi 0.96, ta 0.95,
# kn 0.92, te 0.84, ml 0.63). It is NOT used for auto-detection: left to guess,
# it labelled Telugu/Kannada/Malayalam speech as Tamil or Gujarati.
WHISPER_FAST_MODEL = WHISPER_MODEL_DIR / "large-v3-turbo-ct2"
# CUDA measured ~5x faster than CPU for this model/hardware (2026-10-10:
# ~1.6s vs ~8-9s for an 8s clip) once the nvidia-cublas-cu12/nvidia-cudnn-cu12
# pip packages were installed -- see audio/asr.py for why PATH (not
# os.add_dll_directory) was needed to make ctranslate2 actually find them.
# transcribe() falls back to CPU automatically if CUDA isn't usable on a
# given machine (e.g. those packages aren't installed there).
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "int8_float16"  # fits beside the LLM in 6 GB VRAM

PIPER_VOICE_DIR = ROOT / "models" / "piper"
PIPER_SAMPLE_RATE = 22050

# Piper covers English/Hindi/Telugu/Malayalam but has no Tamil or Kannada
# voice at all (checked directly against its full voice list, 2026-10-10).
# facebook/mms-tts-<code> (VITS, via transformers) fills those two gaps --
# both verified working offline after the one-time download. Every voice
# below was checked to actually produce valid, non-silent audio; none has
# been judged for quality/naturalness by a speaker of the language.
TTS_CONFIG = {
    "en": {"engine": "piper", "voice": "en_US-lessac-medium"},
    "hi": {"engine": "piper", "voice": "hi_IN-priyamvada-medium"},
    "te": {"engine": "piper", "voice": "te_IN-maya-medium"},
    "ml": {"engine": "piper", "voice": "ml_IN-meera-medium"},
    "ta": {"engine": "mms", "model": "facebook/mms-tts-tam"},
    "kn": {"engine": "mms", "model": "facebook/mms-tts-kan"},
}
DEFAULT_TTS_LANGUAGE = "en"

AUDIO_SAMPLE_RATE = 16000
