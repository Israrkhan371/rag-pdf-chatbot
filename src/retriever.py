"""
retriever.py
------------
Combines query embedding + Pinecone search, applies similarity threshold filtering.
Stage: Semantic Retrieval
"""

from src.embedder import embed_query
from src.pinecone_client import query_index


def retrieve_context(
    query: str,
    namespace: str,
    top_k: int = 5,
    similarity_threshold: float = 0.3,
    doc_filter: str | None = None,
    page_filter: int | None = None,
) -> list[dict]:
    """
    Retrieve the top-k relevant chunks for a query, filtered by a minimum
    cosine similarity score.

    Returns a list of dicts:
        {"text", "page", "doc_name", "chunk_id", "score"}
    sorted by descending score. Empty list if nothing clears the threshold.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query_vector = embed_query(query)
    matches = query_index(
        query_vector=query_vector,
        namespace=namespace,
        top_k=top_k,
        doc_filter=doc_filter,
        page_filter=page_filter,
    )

    results = []
    for match in matches:
        score = match.get("score", 0.0)
        if score < similarity_threshold:
            continue
        meta = match.get("metadata", {})
        results.append({
            "text": meta.get("text", ""),
            "page": meta.get("page"),
            "doc_name": meta.get("doc_name"),
            "chunk_id": meta.get("chunk_id"),
            "score": round(float(score), 4),
        })

    return results
