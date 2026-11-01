"""Chunk per-page text into overlapping word-window passages, keeping page refs."""

import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    disaster_category: str
    page: int  # 1-indexed
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_document(
    document_id: str,
    disaster_category: str | None,
    pages: list[str],
    chunk_word_size: int,
    chunk_word_overlap: int,
    page_ranges: list[dict] | None = None,
) -> list[Chunk]:
    """Chunk each page independently (chunks never span pages, so the page
    reference stays exact). Pages with too little text (covers, blank
    separators, roman-numeral front matter) are skipped.

    If `page_ranges` is given (a list of {start_page, end_page, category},
    1-indexed and inclusive), only pages falling in one of those ranges are
    chunked, tagged with that range's category -- this is for multi-topic
    source documents where a single disaster_category for the whole file
    would be wrong. Otherwise every page is chunked under `disaster_category`.
    """
    chunks: list[Chunk] = []
    for page_idx, raw_text in enumerate(pages, start=1):
        if page_ranges is not None:
            category = next(
                (
                    r["category"]
                    for r in page_ranges
                    if r["start_page"] <= page_idx <= r["end_page"]
                ),
                None,
            )
            if category is None:
                continue
        else:
            category = disaster_category

        text = _clean(raw_text)
        words = text.split(" ")
        if len(words) < 15:
            continue

        step = chunk_word_size - chunk_word_overlap
        start = 0
        chunk_num = 0
        while start < len(words):
            window = words[start : start + chunk_word_size]
            if len(window) < 15:
                break
            chunk_id = f"{document_id}_p{page_idx}_c{chunk_num}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    disaster_category=category,
                    page=page_idx,
                    text=" ".join(window),
                )
            )
            chunk_num += 1
            start += step
    return chunks
