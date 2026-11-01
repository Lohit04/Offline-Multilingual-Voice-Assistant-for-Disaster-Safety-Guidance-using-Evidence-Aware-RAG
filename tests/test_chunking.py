from disaster_assistant.knowledge.chunking import chunk_document


def _page(n_words: int, word: str = "word") -> str:
    return " ".join(f"{word}{i}" for i in range(n_words))


def test_single_category_chunks_all_pages():
    pages = [_page(20), _page(20)]
    chunks = chunk_document("doc1", "flood", pages, chunk_word_size=15, chunk_word_overlap=3)
    assert all(c.disaster_category == "flood" for c in chunks)
    assert {c.page for c in chunks} == {1, 2}


def test_too_short_pages_are_skipped():
    pages = [_page(20), _page(5)]  # second page below the 15-word minimum
    chunks = chunk_document("doc1", "flood", pages, chunk_word_size=15, chunk_word_overlap=3)
    assert all(c.page != 2 for c in chunks)


def test_page_ranges_only_keep_and_tag_matching_pages():
    # Regression test for the multi-topic document feature (2026-10-10):
    # a single PDF covering several hazards must only contribute chunks
    # for the pages explicitly in scope, tagged with that range's category.
    pages = [_page(20) for _ in range(5)]  # pages 1-5
    page_ranges = [
        {"start_page": 1, "end_page": 2, "category": "cyclone"},
        {"start_page": 4, "end_page": 4, "category": "earthquake"},
        # page 3 and 5 intentionally not covered -- should be dropped entirely
    ]
    chunks = chunk_document(
        "doc1", disaster_category=None, pages=pages, chunk_word_size=15, chunk_word_overlap=3, page_ranges=page_ranges
    )
    pages_seen = {c.page for c in chunks}
    assert pages_seen == {1, 2, 4}
    assert all(c.disaster_category == "cyclone" for c in chunks if c.page in (1, 2))
    assert all(c.disaster_category == "earthquake" for c in chunks if c.page == 4)


def test_chunk_ids_are_unique():
    pages = [_page(40)]
    chunks = chunk_document("doc1", "flood", pages, chunk_word_size=15, chunk_word_overlap=3)
    assert len(chunks) > 1  # otherwise this test isn't exercising uniqueness across multiple chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
