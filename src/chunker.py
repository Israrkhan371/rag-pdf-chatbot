"""
chunker.py
----------
Splits page-level text into overlapping chunks suitable for embedding.
Stage: Text Chunking

Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
paragraph breaks, then sentences, then words -- this produces more coherent
chunks than a blind fixed-size word window, since it avoids cutting
sentences in half wherever possible.
"""

import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _chunk_id(doc_name: str, page: int, index: int) -> str:
    """Generate a stable, unique chunk ID."""
    raw = f"{doc_name}-p{page}-c{index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def chunk_pages(
    pages: list[dict],
    doc_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[dict]:
    """
    Chunk extracted page text using LangChain's RecursiveCharacterTextSplitter.

    Args:
        pages: [{"page": int, "text": str}, ...] from pdf_loader
        doc_name: source document filename, stored in metadata
        chunk_size: target chunk size in characters (≈ chars, not words --
                    RecursiveCharacterTextSplitter operates on characters)
        chunk_overlap: number of overlapping characters between consecutive chunks

    Returns:
        List of chunk dicts: {"id", "text", "page", "doc_name", "chunk_index"}
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph -> sentence -> word -> char
    )

    chunks = []
    global_index = 0

    for page_data in pages: 
        text = page_data["text"]
        if not text.strip():
            continue

        page_splits = splitter.split_text(text)

        for page_chunk_idx, chunk_text in enumerate(page_splits):
            chunks.append({
                "id": _chunk_id(doc_name, page_data["page"], global_index),
                "text": chunk_text,
                "page": page_data["page"],
                "doc_name": doc_name,
                "chunk_index": page_chunk_idx,
            })
            global_index += 1

    return chunks
