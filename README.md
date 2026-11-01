# Offline Multilingual Voice Assistant for Disaster Safety Guidance using Evidence-Aware RAG

An offline, voice-first assistant that answers disaster-safety questions in **English, Hindi, Tamil, Telugu, Kannada and Malayalam**, using official guidance from India's National Disaster Management Authority (NDMA). It needs no internet: it runs on a laptop served over local Wi-Fi, or fully on an Android phone in airplane mode.

During floods and cyclones, networks fail, and general chatbots can invent shelters, routes or phone numbers. This assistant answers only when the official text supports the answer. Otherwise it refuses.

## How it works
- Retrieval-Augmented Generation over 4 NDMA documents (493 pages, 1,322 cited passages)
- Hybrid retrieval: BM25 plus multilingual-e5-small (FAISS)
- Evidence-aware abstention: a retrieval-score floor, an LLM yes/no evidence check, a rule for vague questions, and a live-information guard that never claims to know current shelters, orders or weather
- Evidence-gated instant answers: 51 pre-verified situations, matched in a median of 0.07 s (down from 7.5 s)
- Safe refusal with emergency numbers 112 / 1070 / 1077

## Models
- **Web (laptop server):** faster-whisper (large-v3, large-v3-turbo), Gemma 4 (e4b) via Ollama, Piper and MMS-TTS voices
- **Android (standalone, `android-app/`):** Gemma-3 1B (int4) via MediaPipe, BM25 retrieval, Android on-device speech recognition (English, Hindi) and text-to-speech

## Results (final system)
- 120/120 correct behaviour on a 6-language test set
- 85/87 (97.7%) on a held-out set
- Every instant answer grounded in the right hazard; 48/48 off-topic questions refused
- 77 automated tests

Details: [docs/implementation_status.md](docs/implementation_status.md), [docs/architecture.md](docs/architecture.md); raw results in `experiments/results/`.

## Limitations
The evaluation questions were written by an AI assistant, and native speakers have not yet reviewed the translations. Voice tests used synthetic speech. Phone voice input supports English and Hindi only. The app gives general guidance; it does not replace emergency services or official live alerts.

Final-year M.Tech (Data Science) project, CSI4007 Natural Language Processing, VIT, by G. Lohit, Cattamanchi Videesha and N. Sudeepthi.

## Quick start (demo day)

One-time setup (already done on the dev laptop):
1. `pip install -e .`
2. Install [Ollama](https://ollama.com) and `ollama pull gemma4:e4b`.
3. Build the knowledge base: `python scripts/ingest_documents.py` (needs internet once).
4. Right-click `scripts\open_firewall.ps1` -> *Run with PowerShell* -> accept the admin prompt (lets phones reach the laptop; Windows blocks this by default on Wi-Fi marked "Public").
5. For the phone's Voice tab only: create the HTTPS certificate (see "Using it from a phone" below).

Every time:
1. Make sure Ollama is running (it normally starts with Windows; otherwise run `ollama serve`).
2. Double-click **`start_demo.bat`**. It first stops any leftover copy of the server (so ports are never "already in use"), runs `scripts/health_check.py` (indexes, Ollama + model, voice models, cert, ports) and then starts the server. Wait ~30 s for model warm-up. The status pill at the top of the page shows **Ready** when everything works, or tells you what is missing (e.g. "Server is up, but Ollama is not running").
3. The console prints the URLs, e.g. `https://172.17.76.13:8443`. Open that on the laptop (`https://localhost:8443`) or on any phone on the same network (laptop hotspot, phone hotspot or shared Wi-Fi). Accept the one-time self-signed-certificate warning.
4. Text works over plain HTTP too (`http://<ip>:8001`); the **Voice** tab needs the HTTPS URL because browsers only allow the microphone on secure origins.

If the laptop is connected to a phone hotspot, its IP changes: restart the server and use the newly printed URL.

Tests: `python -m pytest tests -q` (about 2 minutes; needs Ollama running for the API tests).
Evaluation: `python scripts/evaluate_pipeline.py` (120 question/language runs, results in `experiments/results/`), then `python scripts/summarize_eval.py <results.jsonl>`. Robustness: `python scripts/stress_api.py` against a running server.

## Repo layout
- `src/disaster_assistant/` — application code: `knowledge/` (ingestion), `retrieval/` (hybrid BM25+dense), `generation/` (grounded answers + Ollama client), `safety/` (abstention, live-info guard), `audio/` (ASR, TTS, mic input), `api/app.py` (Streamlit UI, desktop-only), `api/server.py` + `api/static/` (FastAPI + mobile-friendly web app, reachable from a phone), `pipeline.py` (shared voice sequencing).
- `data/`, `indexes/`, `models/` — local artifacts, not committed (see `.gitignore`).
- `docs/` — living project documentation (architecture, literature review, research gap, experiment protocol, implementation status).
- `scripts/`, `experiments/`, `tests/` — ingestion/eval scripts, experiment configs/results, test suite.

## Setup
`pip install -e .` installs the pinned dependencies (see `pyproject.toml` for a note on the `av<19` pin, required for faster-whisper). Requires a local [Ollama](https://ollama.com) with a pulled model (`OLLAMA_MODEL` in `src/disaster_assistant/config.py`) and internet access only for the one-time ingestion step (`python scripts/ingest_documents.py`) and model downloads — normal question-answering afterward does not call out.

Try it:
- Text: `python scripts/ask.py "What should I do if floodwater enters my house?"` (add `--lang ta`/`hi`/`te`/`kn`/`ml` for a different answer language)
- Voice CLI: `python scripts/voice_ask.py --wav <file>` or `--mic <seconds>`
- Desktop voice/text UI: `streamlit run src/disaster_assistant/api/app.py --server.address localhost`

### Using it from a phone (same Wi-Fi, no internet needed)
In this mode the phone is a *client* to this machine, which does the actual work. (The separate Android app in `android-app/` runs fully on the phone instead.) Both the phone and this machine need to be on the same Wi-Fi network; no internet is required.

1. One-time: generate a local HTTPS certificate (needed for the browser to allow microphone access) --
   ```
   mkdir certs
   openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/dev.key -out certs/dev.crt -days 365 -subj "/CN=disaster-assistant-local" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:<this machine's LAN IP>"
   ```
   Find your LAN IP with `ipconfig` (Windows) and look for the Wi-Fi adapter's IPv4 address. The server also prints it at startup. A certificate generated for one IP still works on another network, the browser just shows its warning again.
2. Start the server: double-click `start_demo.bat` (or `python -m disaster_assistant.api.server` with `PYTHONPATH=src`)
3. On your phone's browser, go to `https://<this machine's LAN IP>:8443` — accept the one-time "not trusted" certificate warning (it's self-signed, expected for a local dev setup).
4. Optional: use the browser's "Add to Home Screen" option to install it like an app.

If the phone can't reach the page at all, check: both devices on the *same* Wi-Fi network, and Windows Firewall prompting to allow `python.exe` on first run (allow it).

## License
Code: [MIT](LICENSE). The NDMA guidance text and the third-party models (Whisper, Gemma, multilingual-e5, Piper, MMS-TTS) remain under their own owners' terms.
