"""
embedder.py
-----------
Wraps a Sentence Transformer model via LangChain's HuggingFaceEmbeddings,
so the rest of the pipeline (and PineconeVectorStore) talks to a standard
LangChain Embeddings interface (.embed_documents / .embed_query).
Stage: Embedding Generation
"""

import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import streamlit as st
    _cache_decorator = st.cache_resource
except Exception:
    # Fallback for non-Streamlit contexts (e.g. tests, scripts)
    def _cache_decorator(func):
        return func

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@_cache_decorator
@lru_cache(maxsize=1)
def get_embedder(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """
    Load and cache a LangChain Embeddings object backed by Sentence-Transformers.
    Cached with st.cache_resource so the model is loaded once per Streamlit
    server process, not once per script rerun (Streamlit reruns the whole
    script on every interaction).
    normalize_embeddings=True makes dot-product equivalent to cosine similarity,
    matching the cosine metric configured on the Pinecone index.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """Embed a batch of texts (e.g. document chunks). Returns a list of float vectors."""
    embedder = get_embedder(model_name)
    return embedder.embed_documents(texts)


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    """Embed a single query string."""
    embedder = get_embedder(model_name)
    return embedder.embed_query(query)


def get_embedding_dimension(model_name: str = DEFAULT_MODEL) -> int:
    """Return the vector dimension for the given model (needed for Pinecone index creation)."""
    # SentenceTransformer is used directly here only to read the dimension
    # cheaply, without re-embedding anything through the LangChain wrapper.
    return SentenceTransformer(model_name).get_sentence_embedding_dimension()
