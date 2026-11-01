"""Text-only CLI for testing the retrieval+generation+abstention pipeline.

Usage:
  python scripts/ask.py "What should I do if floodwater enters my house?"
  python scripts/ask.py --lang ta "<question in Tamil>"
"""

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disaster_assistant.retrieval.hybrid import HybridRetriever
from disaster_assistant.generation.answer import answer_question
from disaster_assistant.generation.ollama_client import OllamaError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--lang", default="auto", help="answer language code (en/hi/ta/te/kn/ml), or 'auto' to match the question's own language")
    args = parser.parse_args()

    print("Loading retriever (BM25 + FAISS + embedding model) ...")
    try:
        retriever = HybridRetriever()
    except FileNotFoundError as e:
        print(f"Index files missing ({e}). Run scripts/ingest_documents.py first.")
        sys.exit(1)

    t0 = time.time()
    try:
        result = answer_question(retriever, args.question, language=args.lang)
    except OllamaError as e:
        print(f"Local LLM call failed ({e}). Is Ollama running?")
        sys.exit(1)
    elapsed = time.time() - t0

    print(f"\nQ: {args.question}")
    print(f"A: {result.text}")
    print(
        f"\n[abstained={result.abstained} reason={result.abstention_reason} "
        f"answer_language={result.answer_language} latency={elapsed:.1f}s]"
    )
    if result.sources:
        print("Sources:")
        for s in result.sources:
            print(f"  - {s.document_id} p{s.page} (dense={s.dense_score:.3f}, bm25={s.bm25_score:.2f})")


if __name__ == "__main__":
    main()
