"""Full voice pipeline: audio in -> ASR -> retrieval+generation -> TTS -> audio out.

Usage:
  python scripts/voice_ask.py --mic 8                  # record 8s from the microphone
  python scripts/voice_ask.py --wav path/to/question.wav
"""

import argparse
import sys
import time
from pathlib import Path

# Windows consoles default to a legacy codepage (e.g. cp1252) that can't
# print Tamil (or most non-Latin) text -- reconfigure to UTF-8 so this
# doesn't crash on a print() call. No effect on non-Windows/UTF-8 terminals.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disaster_assistant import config
from disaster_assistant.audio.input import record_from_mic
from disaster_assistant.retrieval.hybrid import HybridRetriever
from disaster_assistant.generation.ollama_client import OllamaError
from disaster_assistant.pipeline import process_voice_question, EmptyTranscriptionError


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mic", type=float, help="record N seconds from the microphone")
    group.add_argument("--wav", type=str, help="path to a pre-recorded WAV question")
    parser.add_argument("--out", type=str, default=str(config.DATA_PROCESSED / "voice_ask_response.wav"))
    args = parser.parse_args()

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if args.mic:
        question_wav = str(config.DATA_PROCESSED / "voice_ask_question.wav")
        print(f"Recording {args.mic}s from the microphone ...")
        try:
            record_from_mic(args.mic, question_wav)
        except Exception as e:
            print(f"Microphone recording failed ({e}). Is a microphone connected and not in use by another app?")
            sys.exit(1)
    else:
        question_wav = args.wav
        if not Path(question_wav).exists():
            print(f"WAV file not found: {question_wav}")
            sys.exit(1)

    print("Loading retriever (BM25 + FAISS + embedding model) ...")
    t_load = time.time()
    try:
        retriever = HybridRetriever()
    except FileNotFoundError as e:
        print(f"Index files missing ({e}). Run scripts/ingest_documents.py first.")
        sys.exit(1)
    load_latency = time.time() - t_load

    print("Transcribing, retrieving, answering, and synthesizing speech ...")
    try:
        result = process_voice_question(retriever, question_wav, args.out)
    except EmptyTranscriptionError as e:
        print(str(e))
        return
    except OllamaError as e:
        print(f"Local LLM call failed ({e}). Is Ollama running ({config.OLLAMA_HOST})?")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed ({e}).")
        sys.exit(1)

    print(f"\nTranscript ({result.detected_language}, p={result.language_probability:.2f}): {result.transcript}")
    for w in result.warnings:
        print(f"  [warning] {w}")
    status = f"ABSTAINED: {result.answer.abstention_reason}" if result.answer.abstained else "grounded"
    print(f"\nAnswer ({status}, language={result.answer_language}):")
    print(f"  {result.answer.text}")
    print(f"\nSpeech written to {args.out}")
    t = result.timings
    print(
        f"[latency] retriever_load={load_latency:.1f}s asr={t['asr']:.1f}s "
        f"answer={t['answer']:.1f}s tts={t['tts']:.1f}s pipeline_total={t['total']:.1f}s "
        f"(retriever_load only happens once per script run, not per question)"
    )


if __name__ == "__main__":
    main()
