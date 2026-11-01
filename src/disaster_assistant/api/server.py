"""FastAPI backend for the web/mobile-browser frontend.

Run with: python -m disaster_assistant.api.server
Serves the pipeline over HTTPS on the local network (phones on the same
Wi-Fi can reach it at https://<this machine's LAN IP>:8443) -- HTTPS is
required for the Voice tab, since browsers only allow microphone access
(getUserMedia) from a "secure context" (HTTPS or localhost), not plain
HTTP to a LAN IP. The cert is self-signed (certs/dev.*, gitignored,
regenerate per machine) -- phones will show a one-time "not trusted"
warning to click through; this is a local dev setup, not a publicly
trusted deployment.
"""

import base64
import functools
import http.server
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import requests
import uvicorn

from disaster_assistant import config
from disaster_assistant.retrieval.hybrid import HybridRetriever
from disaster_assistant.generation.answer import answer_question
from disaster_assistant.generation import ollama_client
from disaster_assistant.generation.ollama_client import OllamaError
from disaster_assistant.audio.tts import synthesize_to_wav
from disaster_assistant.pipeline import process_voice_question, EmptyTranscriptionError

STATIC_DIR = Path(__file__).resolve().parent / "static"
CERTS_DIR = config.ROOT / "certs"

app = FastAPI(title="Disaster Safety Assistant")
# The bundled Android app loads its UI from file:// (see android-app/), so
# its fetch() calls to this API arrive as cross-origin requests with
# Origin: null. Allowing all origins is a deliberate, scoped trade-off
# here, not a general practice -- this server is only ever reachable on a
# local Wi-Fi network you control, never the public internet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> HybridRetriever:
    # FastAPI runs sync endpoints in a thread pool, and the warm-up below
    # runs in its own thread -- without this lock, two threads can both see
    # _retriever as None and both construct a HybridRetriever (loading the
    # embedding model twice). Confirmed directly: the server log showed
    # "Loading weights" twice for what should have been one load, and the
    # first real request after startup was just as slow as a cold one
    # despite the warm-up having already run.
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = HybridRetriever()
    return _retriever


@app.get("/api/health")
def health() -> JSONResponse:
    """Cheap self-diagnosis for the UI's status pill: the web server being up
    does not mean questions will work -- they need Ollama and its model."""
    ollama_up, model_present = False, False
    try:
        tags = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=2).json()
        ollama_up = True
        model_present = any(m.get("name") == config.OLLAMA_MODEL for m in tags.get("models", []))
    except Exception:
        pass
    return JSONResponse({"server": "ok", "ollama": ollama_up, "model_present": model_present,
                         "model": config.OLLAMA_MODEL, "retriever_loaded": _retriever is not None})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


def _answer_to_dict(answer, extra: dict | None = None) -> dict:
    d = {
        "text": answer.text,
        "abstained": answer.abstained,
        "abstention_reason": answer.abstention_reason,
        "answer_language": answer.answer_language,
        "search_question": answer.search_question,
        "answer_timings": answer.timings,
        "cached": answer.cached,
        "sources": [
            {
                "document_id": s.document_id,
                "page": s.page,
                "disaster_category": s.disaster_category,
                "dense_score": s.dense_score,
                "bm25_score": s.bm25_score,
                "text": s.text,
            }
            for s in answer.sources
        ],
    }
    if extra:
        d.update(extra)
    return d


@app.post("/api/ask")
def ask(question: str = Form(...), language: str = Form("auto")) -> JSONResponse:
    if not question.strip():
        return JSONResponse({"error": "Please type or say a question first."}, status_code=400)
    started = time.perf_counter()
    try:
        answer = answer_question(get_retriever(), question, language=language)
    except OllamaError as e:
        return JSONResponse({"error": f"Local LLM call failed: {e}"}, status_code=503)
    except FileNotFoundError as e:
        return JSONResponse({"error": f"Index files missing: {e}"}, status_code=500)
    return JSONResponse(_answer_to_dict(answer, extra={"elapsed_s": round(time.perf_counter() - started, 2)}))


def _answer_cache():
    from disaster_assistant.retrieval.answer_cache import get_answer_cache

    return get_answer_cache(get_retriever().embedder)


@app.get("/api/instant_ids")
def instant_ids() -> JSONResponse:
    """Which prepared situations exist (e.g. "flood.during"), so the home
    screen's hazard tiles only offer Before/During/After where an answer exists."""
    cache = _answer_cache()
    return JSONResponse({"ids": [i["id"] for i in cache.intents] if cache else []})


@app.get("/api/instant")
def instant(id: str, language: str = "en") -> JSONResponse:
    """A prepared situation's stored, cited answer by id -- the hazard tiles.
    No matching and no LLM: the person picked the situation themselves."""
    cache = _answer_cache()
    if cache is None:
        return JSONResponse({"error": "Instant answers are not available."}, status_code=503)
    intent = next((i for i in cache.intents if i["id"] == id), None)
    if intent is None:
        return JSONResponse({"error": "Unknown situation."}, status_code=404)
    lang = language if language in intent["answers"] else "en"
    return JSONResponse({
        "text": intent["answers"][lang], "answer_language": lang, "abstained": False, "cached": True,
        "situation": id,
        "sources": [{**s, "dense_score": 0.0, "bm25_score": 0.0} for s in intent["sources"]],
    })


@app.post("/api/tts")
def tts(text: str = Form(...), language: str = Form("en")) -> JSONResponse:
    """Speech for an answer already shown on screen (fetched separately so the
    text never waits on synthesis). Runs in the thread pool (sync endpoint)."""
    text = text.strip()
    if not text:
        return JSONResponse({"error": "Nothing to read aloud."}, status_code=400)
    if language not in config.TTS_CONFIG:
        return JSONResponse({"error": f"No voice for language '{language}'."}, status_code=400)
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = str(Path(tmp_dir) / "speech.wav")
            synthesize_to_wav(text[:3000], out_path, language=language)
            audio_b64 = base64.b64encode(Path(out_path).read_bytes()).decode("ascii")
    except Exception as e:
        print(f"tts failed: {type(e).__name__}: {e}")
        return JSONResponse({"error": "Could not generate audio for this answer."}, status_code=500)
    return JSONResponse({"audio_base64": audio_b64, "seconds": round(time.perf_counter() - started, 1)})


@app.post("/api/voice_ask")
async def voice_ask(
    audio: UploadFile = File(...), with_audio: bool = Form(True), language: str = Form("auto")
) -> JSONResponse:
    # A chosen language takes the fast speech-recognition path (see config.WHISPER_FAST_MODEL).
    spoken = language if language in config.TTS_CONFIG else None
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = str(Path(tmp_dir) / "question.audio")
        out_path = str(Path(tmp_dir) / "answer.wav")
        with open(in_path, "wb") as f:
            f.write(await audio.read())

        try:
            result = process_voice_question(get_retriever(), in_path, out_path if with_audio else None, language=spoken)
        except EmptyTranscriptionError as e:
            return JSONResponse({"error": str(e), "empty_transcription": True}, status_code=200)
        except OllamaError as e:
            return JSONResponse({"error": f"Local LLM call failed: {e}"}, status_code=503)
        except FileNotFoundError as e:
            return JSONResponse({"error": f"Index files missing: {e}"}, status_code=500)
        except Exception as e:  # undecodable / empty audio, TTS failure, etc.
            # Previously an empty or corrupt upload surfaced as a bare HTTP 500
            # (stress test, 2026-10-13); answer with a usable message instead.
            print(f"voice_ask failed: {type(e).__name__}: {e}")
            return JSONResponse(
                {"error": "Could not process that recording. Please try again, closer to the microphone.",
                 "empty_transcription": True},
                status_code=200,
            )

        audio_b64 = None
        if with_audio:
            with open(out_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("ascii")

    return JSONResponse(
        _answer_to_dict(
            result.answer,
            extra={
                "transcript": result.transcript,
                "detected_language": result.detected_language,
                "language_probability": result.language_probability,
                "answer_language": result.answer_language,
                "warnings": result.warnings,
                "timings": result.timings,
                "audio_base64": audio_b64,
            },
        )
    )


DOWNLOAD_PORT = 8080


class _DownloadHandler(http.server.SimpleHTTPRequestHandler):
    """Serves api/static/ over plain HTTP, specifically so the APK can be
    downloaded. Why a separate plain-HTTP server at all: Android's system
    Download Manager (which actually fetches files, as opposed to the
    browser's own page-rendering engine) does its own TLS validation and
    does NOT inherit the "proceed anyway" exception a user grants Chrome
    for this server's self-signed cert -- downloads over the HTTPS port
    were failing silently for exactly this reason. Plain HTTP has no cert
    to distrust. Only the APK needs this; page rendering and the Voice
    tab's microphone access still require the HTTPS port.
    """

    def guess_type(self, path):
        # Overriding this (rather than adding a second Content-Type header
        # in end_headers) avoids sending two Content-Type headers for the
        # same response -- found by actually inspecting the response with
        # curl -I, not assumed: the base class already sets one via its own
        # guess_type() before end_headers() runs.
        if str(path).endswith(".apk"):
            return "application/vnd.android.package-archive"
        return super().guess_type(path)

    def end_headers(self):
        if self.path.endswith(".apk"):
            self.send_header("Content-Disposition", 'attachment; filename="disaster-assistant.apk"')
        super().end_headers()


def _start_download_server() -> None:
    handler = functools.partial(_DownloadHandler, directory=str(STATIC_DIR))
    server = http.server.ThreadingHTTPServer(("0.0.0.0", DOWNLOAD_PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"APK download server (plain HTTP, for Android's Download Manager) on :{DOWNLOAD_PORT}")


def _warm_up() -> None:
    """Loads the retriever (~9-10s: embedding model) and sends a throwaway
    Ollama call (~30s cold / loads the model into VRAM) in the background at
    server startup, so the first real question of a session doesn't pay
    either cost -- found directly (2026-10-10) that Ollama's cold load is
    the single largest remaining per-request latency item once it happens.
    Runs in a thread so server startup itself isn't delayed by this.
    """
    try:
        retriever = get_retriever()
        from disaster_assistant.retrieval.answer_cache import get_answer_cache

        if get_answer_cache(retriever.embedder) is not None:
            print("Warm-up: instant-answer cache loaded.")
    except Exception as e:
        print(f"Warm-up: retriever preload failed ({e}) -- will retry on first real request.")
    try:
        ollama_client.generate("Reply with OK.")
    except Exception as e:
        print(f"Warm-up: Ollama ping failed ({e}) -- is it running? Will retry on first real request.")
    _warm_up_speech()


# One short word per language, in its own script (MMS voices need native text).
_WARMUP_PHRASES = {
    "en": "flood safety", "hi": "बाढ़", "ta": "வெள்ளம்",
    "te": "వరద", "kn": "ಪ್ರವಾಹ", "ml": "വെള്ളപ്പൊക്കം",
}


def _warm_up_speech() -> None:
    """Loads the speech-recognition model and every language's voice before the
    first request. Measured 2026-10-13: the first spoken question took ~35 s
    (18 s of it loading Whisper large-v3, the rest loading voices) versus
    ~12-25 s once warm."""
    try:
        from disaster_assistant.audio.asr import transcribe

        with tempfile.TemporaryDirectory() as tmp_dir:
            clip = str(Path(tmp_dir) / "warm.wav")
            for lang, phrase in _WARMUP_PHRASES.items():
                if lang in config.TTS_CONFIG:
                    synthesize_to_wav(phrase, clip, lang)
            synthesize_to_wav(_WARMUP_PHRASES["en"], clip, "en")  # last clip = English, used to warm ASR
            # Fast model only: it serves every question with a chosen language.
            # large-v3 (auto-detect) loads on first use -- both resident beside
            # the LLM would nearly fill the 6 GB GPU.
            transcribe(clip, language="en")
        print("Warm-up: speech recognition and all voices loaded.")
    except Exception as e:
        print(f"Warm-up: speech preload failed ({type(e).__name__}: {e}) -- will load on first voice request.")


PLAIN_API_PORT = 8001


def _start_plain_api_server() -> None:
    """Runs the exact same FastAPI app on a plain-HTTP port, for the
    bundled Android app's data calls specifically.

    Why: the bundled app's page loads from file:// (a context Chromium/
    WebView already treats as secure for getUserMedia -- confirmed this
    doesn't need HTTPS), but its fetch() calls to the HTTPS port were
    failing anyway. Likely the same root cause already confirmed for the
    APK download: Android's SSL error handling for *background* requests
    (fetch/XHR) from a WebView is known to be unreliable across versions,
    separately from `onReceivedSslError`, which mainly covers page loads.
    Plain HTTP has no certificate to distrust, so this sidesteps the
    problem entirely rather than trying to force WebView to trust it.
    The browser path (HTTPS, for microphone access when visited as a
    normal webpage) is unaffected -- this is an additional port, not a
    replacement.
    """
    config_obj = uvicorn.Config(app, host="0.0.0.0", port=PLAIN_API_PORT, log_level="warning")
    server = uvicorn.Server(config_obj)
    threading.Thread(target=server.run, daemon=True).start()
    print(f"Plain-HTTP API server (for the bundled Android app) on :{PLAIN_API_PORT}")


def _lan_ips() -> list[str]:
    """IPv4 addresses of this machine's network interfaces (best effort).
    Printed at startup so the right URL for the phone is visible instead of
    having to run ipconfig: the address differs per network (laptop hotspot
    192.168.137.1, phone hotspot, campus Wi-Fi...)."""
    import socket
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        # UDP "connect" sends nothing; it only asks the OS which interface
        # would route outward, which is the one the phone can usually reach.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def _print_urls(https: bool) -> None:
    scheme, port = ("https", 8443) if https else ("http", 8000)
    print("\n=== Open one of these on the phone (same network as this laptop) ===")
    for ip in _lan_ips():
        print(f"  {scheme}://{ip}:{port}      (APK: http://{ip}:{DOWNLOAD_PORT}/disaster-assistant.apk)")
    print("If the phone can't connect, run scripts/open_firewall.ps1 once (as admin).\n")


def main() -> None:
    _start_download_server()
    _start_plain_api_server()
    threading.Thread(target=_warm_up, daemon=True).start()

    cert_path = CERTS_DIR / "dev.crt"
    key_path = CERTS_DIR / "dev.key"
    if not cert_path.exists() or not key_path.exists():
        print(
            f"Missing TLS cert/key at {CERTS_DIR} -- generate with openssl first "
            "(see docs/architecture.md for the exact command). Falling back to plain HTTP: "
            "the Voice tab's microphone will NOT work from a phone over the LAN (browsers "
            "require HTTPS for that), only the Text tab will."
        )
        _print_urls(https=False)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        _print_urls(https=True)
        uvicorn.run(app, host="0.0.0.0", port=8443, ssl_certfile=str(cert_path), ssl_keyfile=str(key_path))


if __name__ == "__main__":
    main()
