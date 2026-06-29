"""
pinecone_client.py
-------------------
Handles all direct interaction with Pinecone: index creation, namespaces,
upserting vectors, querying, and metadata-based filtering.

Index lifecycle (creation, readiness check, deletion) is managed directly
through the Pinecone SDK so it's explicit and easy to demonstrate for the
assignment. Document upsert/query operations are routed through LangChain's
PineconeVectorStore, which wraps the same underlying index and exposes a
standard LangChain Document-based interface (Document.page_content / .metadata),
including similarity_search_with_score for retrieval.

Stage: Pinecone Vector Indexing + Semantic Retrieval (storage layer)
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from src.embedder import get_embedder

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except Exception:
    def _cache_resource(func):
        return func

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-pdf-chatbot")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")


class PineconeConnectionError(Exception):
    pass


@_cache_resource
def get_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise PineconeConnectionError("PINECONE_API_KEY is not set. Check your .env file.")
    try:
        return Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        raise PineconeConnectionError(f"Failed to connect to Pinecone: {e}")


@_cache_resource
def ensure_index(dimension: int, metric: str = "cosine"):
    """
    Create the index if it doesn't exist yet, then return a raw Pinecone Index handle.
    Cached per (dimension, metric) so the has_index/create_index network round-trip
    only happens once per Streamlit session instead of on every upsert/query call.
    """
    pc = get_client()
    try:
        if not pc.has_index(INDEX_NAME):
            pc.create_index(
                name=INDEX_NAME,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            )
            # Wait until the index is ready
            while not pc.describe_index(INDEX_NAME).status.ready:
                time.sleep(1)
        return pc.Index(INDEX_NAME)
    except PineconeConnectionError:
        raise
    except Exception as e:
        raise PineconeConnectionError(f"Pinecone index setup failed: {e}")


@_cache_resource
def get_vector_store(namespace: str, dimension: int) -> PineconeVectorStore:
    """
    Build a LangChain PineconeVectorStore bound to a specific namespace,
    using the same embedding model as the rest of the pipeline (src/embedder.py)
    so chunk vectors and query vectors are produced identically.
    Cached per namespace so this isn't rebuilt on every query/upsert call.
    """
    index = ensure_index(dimension=dimension)
    embedder = get_embedder()
    return PineconeVectorStore(index=index, embedding=embedder, namespace=namespace, text_key="text")


def upsert_chunks(chunks: list[dict], vectors: list[list[float]], namespace: str):
    """
    Upsert chunk vectors with metadata into a namespace via LangChain's
    PineconeVectorStore.add_documents (uses precomputed embeddings via add_embeddings
    where supported, falling back to passing texts + ids + metadata directly).

    Each chunk dict must have: id, text, page, doc_name, chunk_index
    Metadata stored: doc_name, page, chunk_id (text is stored automatically as
    the LangChain Document's page_content, under the "text" key in Pinecone metadata).
    """
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must be the same length")

    vector_store = get_vector_store(namespace=namespace, dimension=len(vectors[0]))

    documents = [
        Document(
            page_content=chunk["text"][:1000],  # cap stored text to keep metadata light
            metadata={
                "doc_name": chunk["doc_name"],
                "page": chunk["page"],
                "chunk_id": chunk["id"],
            },
        )
        for chunk in chunks
    ]
    ids = [chunk["id"] for chunk in chunks]

    try:
        vector_store.add_documents(documents=documents, ids=ids)
    except Exception as e:
        raise PineconeConnectionError(f"Upsert to Pinecone failed: {e}")

    return len(documents)


def query_index(
    query_vector: list[float],
    namespace: str,
    top_k: int = 5,
    doc_filter: str | None = None,
    page_filter: int | None = None,
):
    """
    Query Pinecone via LangChain's PineconeVectorStore.similarity_search_with_score,
    using a precomputed query embedding so the embedding model is only invoked once
    per query (the vector_store.embed_query path is bypassed by calling the
    underlying index search directly through the vector store's client).

    doc_filter / page_filter apply Pinecone metadata filtering when provided.
    Returns a normalized match list: [{"id", "score", "metadata"}, ...]
    """
    vector_store = get_vector_store(namespace=namespace, dimension=len(query_vector))

    filter_dict = {}
    if doc_filter:
        filter_dict["doc_name"] = {"$eq": doc_filter}
    if page_filter is not None:
        filter_dict["page"] = {"$eq": page_filter}

    try:
        # similarity_search_by_vector_with_score lets us reuse the embedding
        # already computed in retriever.py instead of re-embedding the query text.
        results = vector_store.similarity_search_by_vector_with_score(
            embedding=query_vector,
            k=top_k,
            filter=filter_dict or None,
        )
    except Exception as e:
        raise PineconeConnectionError(f"Pinecone query failed: {e}")

    matches = []
    for doc, score in results:
        matches.append({
            "id": doc.metadata.get("chunk_id"),
            "score": score,
            "metadata": {
                "text": doc.page_content,
                "doc_name": doc.metadata.get("doc_name"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
            },
        })

    return matches


def delete_namespace(namespace: str):
    """Delete all vectors in a namespace (e.g. when removing a document)."""
    index = get_client().Index(INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass  # namespace may not exist yet; not a fatal error
