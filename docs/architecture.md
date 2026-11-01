# Architecture

Status: a working baseline is implemented and manually tested end-to-end (text path and full voice path) as of 2026-10-10 -- see docs/implementation_status.md for exactly what's verified vs. not. This file records the actual implementation and the reasoning behind choices, updated as things change. No literature review or formal evaluation has been done yet (deferred per user instruction); none of the component choices below are claimed to be the *best* choice, only a working, explainable starting point.

## System framing

Offline-capable, multilingual, voice-driven disaster information assistant. Retrieves from a locally curated, traceable knowledge base and answers only when evidence supports it; abstains rather than inventing emergency instructions, shelter locations, or live conditions. See root `CLAUDE.md` §4–6 for the full framing and §14 for safety requirements — this file should not restate those, only record what's been decided and why.

## Pipeline (planned)

```
audio -> preprocessing -> offline ASR -> language ID/normalization -> query understanding
     -> retrieval (lexical + dense [+ reranking]) -> evidence assessment
     -> {grounded generation | abstention/fallback} -> safety checks
     -> response formatting -> offline TTS
```

Separately, a document-ingestion pipeline (not request-time) builds the local indexes from authoritative source documents. Normal question-answering must not require network access or a cloud API call.

## Infrastructure decisions made so far (2026-10-10, during environment setup)

| Decision | Choice | Reasoning |
|---|---|---|
| Python interpreter | System Python 3.14.0 (already default; 3.11.4 also available but not currently needed) | Despite being a newer CPython release, it is ~1 year old at time of setup and the ML ecosystem has caught up: `torch` (2.10.0+cpu), `numpy`, `faiss-cpu`, and `ctranslate2` (faster-whisper's backend) all ship working `cp314` Windows wheels, verified by direct `pip download` probes rather than assumption. |
| Local LLM runtime | **Ollama** (already installed, v0.40.2) rather than `llama-cpp-python` as a Python library | `llama-cpp-python` publishes no prebuilt Windows wheel on PyPI for any tested Python version (3.11 or 3.14) — only an sdist requiring CMake + MSVC build tools, neither installed on this machine. Ollama needs no compiler, is already present, and runs fully offline once a model is pulled (local HTTP API on localhost — does not violate the no-mandatory-cloud-dependency requirement). Trade-off: integration is via local HTTP calls rather than an in-process Python binding; revisit only if that becomes a real limitation (e.g. constrained decoding/grammars). |
| GPU use | **Ollama** uses the GPU (confirmed via `/api/ps`: ~3GB VRAM) via its own bundled runtime. **faster-whisper/ctranslate2** now also uses the GPU (`WHISPER_DEVICE="cuda"`), via the `nvidia-cublas-cu12`/`nvidia-cudnn-cu12`/`nvidia-cuda-runtime-cu12` pip packages (optional extra, `pip install -e ".[gpu]"`) -- the driver alone wasn't enough. The project's own `torch` install remains CPU-only, used by the embedding model and Tamil TTS. | Ollama bundles its own CUDA-capable runtime independent of the Python `torch` install -- these were, and still are, separate GPU-usage questions. ctranslate2's CUDA library loading needed the DLL directories on `PATH` specifically; `os.add_dll_directory` alone (tried first) did not work. |

## Implemented pipeline (updated 2026-10-10, second pass)

```
WAV file or mic (sounddevice / browser via Streamlit) -> faster-whisper "small" (CUDA/float16, falls back to CPU/int8 automatically) ASR
  -> HybridRetriever (BM25 + FAISS/multilingual-e5-small, 0.5/0.5 score fusion)
  -> baseline abstention check (dense-score floor 0.80; BM25 overlap required only when
     query language == corpus language, i.e. English -- see Findings #2 in implementation_status.md)
  -> {grounded Ollama (gemma4:e4b) generation, answer_language-aware prompt, structural
     live-info disclaimer guard | fixed abstention message with sourced phone numbers}
  -> TTS routed by language (6 languages): Piper for English/Hindi/Telugu/Malayalam,
     facebook/mms-tts-tam / facebook/mms-tts-kan (VITS via transformers) for Tamil/Kannada
     (Piper ships no voice at all for either, checked directly)
  -> WAV
```

Entry points: `scripts/ingest_documents.py` (build indexes), `scripts/ask.py` (text, `--lang`), `scripts/voice_ask.py` (voice CLI, `--wav`/`--mic`), `streamlit run src/disaster_assistant/api/app.py` (UI, text + voice tabs). `src/disaster_assistant/pipeline.py` holds the shared ASR->retrieval->generation->TTS sequencing so the CLI and the UI don't duplicate it.

Component choices made so far, each a pragmatic starting point rather than a benchmarked decision:
- **Embedding model**: `intfloat/multilingual-e5-small`. Confirmed (not just assumed) to do usable cross-lingual retrieval: a correctly-transcribed Tamil query retrieves the right English passages and dense scores land in a similar range to equivalent English queries.
- **Chunking**: fixed 180-word windows with 40-word overlap, one chunk never spans a page; supports per-page-range category overrides for a multi-topic source document (tested).
- **Retrieval fusion**: 0.5 BM25 / 0.5 dense, each min-max normalized within its own candidate pool per query -- still unevaluated, CLAUDE.md's warning about arbitrary weighting still applies.
- **Abstention**: dense-score floor (0.80), with the BM25-overlap requirement now scoped to same-language queries only (see implementation_status.md Finding #2 for why the unscoped version was a real bug, not just a quality gap). Still a crude heuristic -- the dense-score-doesn't-separate-relevant-from-unhelpful problem from the first session is unresolved, this only fixed a different, structural issue.
- **Live-info guard**: `safety/live_info.py` -- a keyword-based detector that forces a fixed disclaimer onto the final answer text in code (not just via a prompt instruction) when a question looks like it's asking for current/live information. English-only detection; a coarse heuristic, not a validated classifier.
- **ASR**: faster-whisper "small", now CUDA-accelerated (float16) with an automatic, tested CPU/int8 fallback. GPU cut an 8s clip from ~8-9s to ~1.6-2.6s. "medium" vs. "small" was measured directly on one Tamil utterance (on CPU, before the GPU fix): "medium" fixed a flood/jaggery mis-transcription but took dramatically longer under RAM pressure (minutes vs. ~9s) -- kept "small" as the default; worth re-measuring now that GPU is available, since that comparison predates this fix.
- **TTS**: `config.TTS_CONFIG` routes by language -- Piper for English/Hindi/Telugu/Malayalam, `facebook/mms-tts-tam`/`facebook/mms-tts-kan` (VITS via `transformers`) for Tamil/Kannada (Piper ships no voice at all for either, checked directly against its full voice list both times). All 6 verified to produce valid, non-silent audio; none quality-judged by a native speaker.
- **LLM**: Ollama `gemma4:e4b`, called over the local HTTP API with `"think": false` and a 30-minute `keep_alive`. The `think: false` setting matters a lot, not just a little: `gemma4:e4b` is a reasoning model that otherwise emits a long internal thinking trace (measured: 931 output tokens, ~19.4s) before a short real answer -- disabling it cut a representative grounded answer from ~22s to ~4.6s with no apparent quality loss. Prompt now requests the answer in a specific target language (6 supported) even though the evidence passages are always English -- works for *some* questions in all 5 non-English languages tested, but retrieval quality for the underlying evidence varies by specific concept, not just language (see implementation_status.md Finding #14 -- an earthquake question retrieved the same wrong generic passage in Hindi/Telugu/Kannada/Malayalam, while flood questions retrieved correctly in the same languages). Occasional generation quirks observed (a stray Hindi word in Tamil output once; unprompted English glosses once), not investigated further.
- **Knowledge base scope**: all 18 hazard categories in the Disaster Survival Guide are now ingested (expanded from the original 3 -- flood/cyclone/earthquake -- per user request, 2026-10-10), via wider `page_ranges` in `data/metadata/documents.json`. No new sourcing was needed; the document was already fully downloaded.
- **UI**: Streamlit (`src/disaster_assistant/api/app.py`), chosen over FastAPI for speed of building a usable prototype; its `st.audio_input` widget doubles as the practical way to test live-mic recording interactively, which can't be driven from this automated environment. Verified to boot and serve; not click-tested in a real browser from here.

## Mobile/web deployment decision (2026-10-10)

The user's actual deployment goal turned out to be: install this on a phone, usable fully offline during a disaster. That goal, taken literally (everything running inside the phone's own hardware, zero other devices), is **not compatible with anything built so far**:

| Component as built | Why it doesn't run on a phone |
|---|---|
| Ollama | Desktop/server application, not an Android/iOS app |
| faster-whisper + CUDA | Needs an NVIDIA GPU |
| FAISS | No standard mobile (Android/iOS) build |
| `transformers`/PyTorch (embedding model, MMS-TTS) | Not how inference is normally shipped in a mobile app |
| Streamlit | A Python web-server UI, not a mobile app |

A genuine on-device port would need: a mobile LLM runtime (llama.cpp cross-compiled for ARM, or MLC-LLM, or Google's MediaPipe/AI Edge) running a much smaller model (1-3B class, not 8B, because phone RAM is far more limited than this dev machine's); `whisper.cpp` instead of `faster-whisper`; Piper's ONNX voices likely portable via ONNX Runtime Mobile, but the MMS-TTS Tamil/Kannada voices would need re-exporting to ONNX (unverified whether that's even straightforward); a smaller/mobile-compatible vector search replacing FAISS; and a native Android (Kotlin) or iOS (Swift) app, or a cross-platform framework (Flutter/React Native) bridging to those native runtimes. That is realistically weeks of work in a different skill set, not something compatible with "quickly."

**Decision, given the explicit speed requirement**: built a FastAPI backend (`api/server.py`) and a mobile-responsive static web frontend (`api/static/`) instead, reusing the existing pipeline code unchanged. The phone connects over HTTPS to this machine on the same local Wi-Fi network (no internet needed) and does the recording/display; this machine still does all the actual inference. This is explicitly **"phone as a client to a local server," not "phone running the AI by itself"** -- a different, weaker claim than the original ask, stated plainly rather than blurred. It still satisfies "no internet dependency" (a local Wi-Fi network with no internet uplink works fine), just not "no other hardware at all." HTTPS (self-signed cert, `certs/dev.{crt,key}`, gitignored) is required because browsers only allow microphone access from a secure context -- plain HTTP to a LAN IP is blocked by the browser itself, not a server-side choice.

If a genuine fully-standalone on-device app is wanted later, treat it as a distinct project phase with its own time budget, not an extension of this session's work.

## Not yet decided
- Primary research contribution — pending literature review, deferred per user instruction. The abstention/cross-lingual bugs found and fixed this session (implementation_status.md Findings #2-3) are concrete, evidence-backed material for that discussion whenever it happens, not just a provisional hunch.
- Final geographic/language scope — Tamil TTS is no longer a blocker (resolved via MMS-TTS), but Tamil ASR accuracy at a practical model size still is, per the Finding #4 trade-off above.
