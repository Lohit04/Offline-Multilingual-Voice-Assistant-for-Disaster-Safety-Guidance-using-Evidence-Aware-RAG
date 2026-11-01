"""Pre-flight check: is everything the server needs actually in place?

Run before a demo:  python scripts/health_check.py
Exit code 0 = all required checks passed.
"""

import json
import socket
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disaster_assistant import config  # noqa: E402

results: list[tuple[str, bool, str, bool]] = []  # (name, ok, detail, required)


def check(name: str, ok: bool, detail: str = "", required: bool = True) -> None:
    results.append((name, ok, detail, required))


def main() -> int:
    # Knowledge base + indexes
    check("chunks.jsonl", config.CHUNKS_PATH.exists(), str(config.CHUNKS_PATH))
    check("FAISS index", config.FAISS_INDEX_PATH.exists())
    check("BM25 corpus", config.BM25_CORPUS_PATH.exists())
    if config.CHUNKS_PATH.exists() and config.BM25_CORPUS_PATH.exists():
        n_chunks = sum(1 for _ in config.CHUNKS_PATH.open(encoding="utf-8"))
        n_bm25 = sum(1 for _ in config.BM25_CORPUS_PATH.open(encoding="utf-8"))
        check("index/chunk counts match", n_chunks == n_bm25, f"chunks={n_chunks} bm25={n_bm25}")

    # Ollama + model
    try:
        tags = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models", [])]
        check("Ollama reachable", True, config.OLLAMA_HOST)
        check(f"Ollama model {config.OLLAMA_MODEL}", config.OLLAMA_MODEL in names, ", ".join(names) or "no models")
    except Exception as e:
        check("Ollama reachable", False, f"{type(e).__name__} -- start it with: ollama serve")

    # Voice models (optional for text-only demo)
    check("Whisper model cached", any((config.WHISPER_MODEL_DIR).glob(f"*whisper-{config.WHISPER_MODEL_SIZE}*")),
          config.WHISPER_MODEL_SIZE, required=False)
    for lang, cfg in config.TTS_CONFIG.items():
        if cfg["engine"] == "piper":
            ok = (config.PIPER_VOICE_DIR / f"{cfg['voice']}.onnx").exists()
            check(f"Piper voice [{lang}]", ok, cfg["voice"], required=False)

    # TLS cert for phone microphone access
    check("HTTPS cert (certs/dev.crt)", (ROOT / "certs" / "dev.crt").exists(),
          "needed for the phone's Voice tab; text works without", required=False)

    # Ports free (server not already running)
    for port in (8443, 8001, 8080):
        with socket.socket() as s:
            in_use = s.connect_ex(("127.0.0.1", port)) == 0
        check(f"port {port} free", not in_use, "in use -- is the server already running?", required=False)

    # Verified helplines registry
    reg = json.loads(config.DOCUMENTS_REGISTRY.read_text(encoding="utf-8"))
    check("verified helplines present", {c["phone"] for c in reg["verified_contacts"]} >= {"112", "1070", "1077"})

    failed_required = 0
    for name, ok, detail, required in results:
        mark = "PASS" if ok else ("FAIL" if required else "warn")
        failed_required += (not ok and required)
        print(f"[{mark}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    print("\nREADY" if not failed_required else f"\nNOT READY: {failed_required} required check(s) failed")
    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main())
