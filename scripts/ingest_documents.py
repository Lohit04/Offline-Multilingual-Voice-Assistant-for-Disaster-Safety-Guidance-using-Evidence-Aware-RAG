"""Run the full ingestion pipeline: registered PDFs -> chunks.jsonl -> lexical + dense indexes.

Usage: python scripts/ingest_documents.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disaster_assistant import config
from disaster_assistant.knowledge.extraction import extract_pages
from disaster_assistant.knowledge.chunking import chunk_document
from disaster_assistant.knowledge.indexing import build_lexical_corpus, build_dense_index


def main() -> None:
    registry = json.loads(config.DOCUMENTS_REGISTRY.read_text(encoding="utf-8"))
    all_chunks = []

    for doc in registry["documents"]:
        pdf_path = config.ROOT / doc["file_path"]
        print(f"Extracting {doc['document_id']} ({doc['title']}) from {pdf_path.name} ...")
        pages = extract_pages(pdf_path)
        chunks = chunk_document(
            document_id=doc["document_id"],
            disaster_category=doc["disaster_category"],
            pages=pages,
            chunk_word_size=config.CHUNK_WORD_SIZE,
            chunk_word_overlap=config.CHUNK_WORD_OVERLAP,
            page_ranges=doc.get("page_ranges"),
        )
        print(f"  -> {len(pages)} pages, {len(chunks)} chunks")
        all_chunks.extend(c.to_dict() for c in chunks)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(all_chunks)} total chunks to {config.CHUNKS_PATH}")

    print("Building lexical (BM25) corpus ...")
    build_lexical_corpus(all_chunks)

    print("Building dense (FAISS) index ... (downloads the embedding model on first run)")
    build_dense_index(all_chunks)

    print("Done.")


if __name__ == "__main__":
    main()
