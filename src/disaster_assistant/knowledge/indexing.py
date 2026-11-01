"""Build the lexical (BM25) and dense (FAISS) indexes from chunked passages."""

import json
import string

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from disaster_assistant import config

# A regex word-character class ([^\W_]+) is wrong for Tamil (and other
# scripts that use combining vowel signs / virama): those combining
# codepoints aren't in Python re's \w, so the regex fragments every word
# at each combining mark -- found 2026-10-10 via a failing test
# (tokenize() on "வெள்ளம்" produced several single-codepoint tokens
# instead of one word). Splitting on whitespace and stripping punctuation
# from each token's edges leaves combining-mark sequences intact.
_EDGE_PUNCT = string.punctuation + "‘’“”—–।"


def tokenize(text: str) -> list[str]:
    return [t for t in (tok.strip(_EDGE_PUNCT) for tok in text.lower().split()) if t]


# English function words. "not"/"no" are deliberately NOT here: they carry
# meaning in safety text ("do not enter floodwater"), even though BM25 gains
# little from them.
_STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from as is are was were be been being am
i me my we our you your he she it its they them their this that these those
what which who whom whose when where why how should would could can will shall may might must
do does did done have has had having there here than then so such into onto about over under
again also just very any some all each other
""".split())


def _stem(token: str) -> str:
    """Minimal plural stripping for ASCII words only (floodwaters -> floodwater,
    enters -> enter). Deliberately crude: no linguistic stemmer dependency,
    and non-ASCII tokens (Indian-language scripts) are left untouched."""
    if not token.isascii() or not token.isalpha() or len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("ss", "us", "is")):
        return token
    return token[:-1] if token.endswith("s") else token


def lexical_tokens(text: str) -> list[str]:
    """Tokens for BM25 (index AND query side): tokenize() minus stopwords,
    plus light stemming. Added 2026-10-13 because function words ("what
    should I do if my") were producing the top keyword hits, pulling
    forest-fire/earthquake chunks above flood guidance for flood queries."""
    return [_stem(t) for t in tokenize(text) if t not in _STOPWORDS]


def build_lexical_corpus(chunks: list[dict]) -> None:
    config.INDEX_LEXICAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c["chunk_id"], "tokens": lexical_tokens(c["text"])}) + "\n")


def build_dense_index(chunks: list[dict], model_name: str = config.EMBEDDING_MODEL_NAME) -> None:
    config.INDEX_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    # e5 models expect a "passage: " / "query: " instruction prefix.
    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    chunk_ids = [c["chunk_id"] for c in chunks]
    with open(config.FAISS_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)
