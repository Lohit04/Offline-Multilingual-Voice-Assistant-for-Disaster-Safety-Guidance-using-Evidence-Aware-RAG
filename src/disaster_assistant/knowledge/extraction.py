"""PDF text extraction with page tracking."""

import pypdf


def extract_pages(pdf_path) -> list[str]:
    """Return a list of per-page extracted text, in page order (index 0 = page 1)."""
    reader = pypdf.PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]
