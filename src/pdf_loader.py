"""
pdf_loader.py
-------------
Handles PDF upload validation, text extraction, and basic cleaning.
Stage: PDF Upload -> Text Extraction
"""

import re
from pypdf import PdfReader

MAX_FILE_SIZE_MB = 20


class PDFLoadError(Exception):
    """Raised when a PDF fails validation or extraction."""
    pass


def validate_pdf(file_bytes: bytes, filename: str) -> None:
    """Validate file size and basic PDF signature before processing."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise PDFLoadError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )
    if not file_bytes.startswith(b"%PDF"):
        raise PDFLoadError(f"'{filename}' does not appear to be a valid PDF file.")


def clean_text(raw_text: str) -> str:
    """Remove common PDF extraction artifacts: extra whitespace, broken hyphenation, page noise."""
    text = re.sub(r"-\n", "", raw_text)          # fix hyphenated line breaks
    text = re.sub(r"\n+", " ", text)               # collapse newlines
    text = re.sub(r"\s{2,}", " ", text)            # collapse extra spaces
    text = text.strip()
    return text


def extract_pages(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract text per page from a PDF.

    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    Raises PDFLoadError on failure (corrupt file, encrypted file, etc.)
    """
    validate_pdf(file_bytes, filename)

    try:
        reader = PdfReader(file_bytes if hasattr(file_bytes, "read") else __import__("io").BytesIO(file_bytes))
    except Exception as e:
        raise PDFLoadError(f"Could not open '{filename}': {e}")

    if reader.is_encrypted:
        raise PDFLoadError(f"'{filename}' is password-protected and cannot be processed.")

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page": i, "text": cleaned})

    if not pages:
        raise PDFLoadError(f"No extractable text found in '{filename}'. It may be a scanned/image-only PDF.")

    return pages
