"""
app.py
------
Streamlit UI for the RAG PDF Chatbot. Wires together every pipeline stage:
PDF Upload -> Text Extraction -> Chunking -> Embedding -> Pinecone Indexing
-> Semantic Retrieval -> LLM Generation -> Answer with Source Reference
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pdf_loader import extract_pages, PDFLoadError
from src.chunker import chunk_pages
from src.embedder import embed_texts
from src.pinecone_client import upsert_chunks, PineconeConnectionError, delete_namespace
from src.retriever import retrieve_context
from src.generator import generate_answer, GenerationError
from src.utils import safe_namespace, log_query

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")

# ---------- Session State ----------
if "documents" not in st.session_state:
    st.session_state.documents = {}   # {namespace: {"filename": str, "num_chunks": int}}
if "history" not in st.session_state:
    st.session_state.history = []     # list of {query, answer, sources}

# ---------- Sidebar: Settings ----------
st.sidebar.header("⚙️ Settings")
chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 2000, 800, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap (characters)", 0, 400, 150, step=25)
top_k = st.sidebar.slider("Top-K chunks to retrieve", 1, 10, 5)
similarity_threshold = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.3, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Indexed Documents")
if st.session_state.documents:
    for ns, meta in st.session_state.documents.items():
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"📄 {meta['filename']} ({meta['num_chunks']} chunks)")
        if col2.button("🗑️", key=f"del_{ns}"):
            delete_namespace(ns)
            del st.session_state.documents[ns]
            st.rerun()
else:
    st.sidebar.caption("No documents indexed yet.")

doc_filter_options = ["All documents"] + [m["filename"] for m in st.session_state.documents.values()]
selected_doc = st.sidebar.selectbox("Filter answers by document", doc_filter_options)

page_filter_value = st.sidebar.text_input("Filter by page number (optional)", "")

# ---------- Main: Title ----------
st.title("📄 RAG PDF Chatbot")
st.caption("Ask questions grounded strictly in your uploaded PDF content — Pinecone-powered retrieval, zero hallucination tolerance.")

# ---------- Upload Section ----------
st.subheader("1. Upload a PDF")
uploaded_files = st.file_uploader(
    "Upload one or more PDFs (max 20MB each)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        namespace = safe_namespace(uploaded_file.name)
        if namespace in st.session_state.documents:
            continue  # already indexed this session

        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            try:
                file_bytes = uploaded_file.read()
                pages = extract_pages(file_bytes, uploaded_file.name)
                chunks = chunk_pages(pages, uploaded_file.name, chunk_size, chunk_overlap)
                vectors = embed_texts([c["text"] for c in chunks])
                num_upserted = upsert_chunks(chunks, vectors, namespace)

                st.session_state.documents[namespace] = {
                    "filename": uploaded_file.name,
                    "num_chunks": num_upserted,
                }
                st.success(f"✅ Indexed '{uploaded_file.name}' — {num_upserted} chunks stored in Pinecone.")
            except PDFLoadError as e:
                st.error(f"❌ {e}")
            except PineconeConnectionError as e:
                st.error(f"❌ Pinecone error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

# ---------- Query Section ----------
st.subheader("2. Ask a Question")
query = st.text_input("Your question about the document(s):")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not query or not query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    elif not st.session_state.documents:
        st.warning("⚠️ Please upload and index at least one PDF first.")
    else:
        doc_filter = None
        if selected_doc != "All documents":
            for ns, meta in st.session_state.documents.items():
                if meta["filename"] == selected_doc:
                    doc_filter = meta["filename"]

        page_filter = None
        if page_filter_value.strip().isdigit():
            page_filter = int(page_filter_value.strip())

        with st.spinner("Retrieving relevant context..."):
            try:
                # Query across all indexed namespaces (multi-document support)
                all_results = []
                target_namespaces = (
                    [ns for ns in st.session_state.documents if st.session_state.documents[ns]["filename"] == doc_filter]
                    if doc_filter else list(st.session_state.documents.keys())
                )
                for ns in target_namespaces:
                    results = retrieve_context(
                        query=query,
                        namespace=ns,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        page_filter=page_filter,
                    )
                    all_results.extend(results)

                all_results.sort(key=lambda r: r["score"], reverse=True)
                all_results = all_results[:top_k]

            except PineconeConnectionError as e:
                st.error(f"❌ Pinecone error: {e}")
                all_results = None
            except ValueError as e:
                st.warning(f"⚠️ {e}")
                all_results = None

        if all_results is not None:
            with st.spinner("Generating answer..."):
                try:
                    result = generate_answer(query, all_results)
                    answer_found = bool(all_results)

                    st.session_state.history.append({
                        "query": query,
                        "answer": result["answer"],
                        "sources": result["sources"],
                    })
                    log_query(doc_filter or "all", query, len(all_results), answer_found)

                except GenerationError as e:
                    st.error(f"❌ {e}")

# ---------- Latest Answer Display ----------
if st.session_state.history:
    latest = st.session_state.history[-1]
    st.subheader("3. Answer")
    st.markdown(f"**{latest['answer']}**")

    if latest["sources"]:
        st.subheader("📎 Source Attribution")
        for i, src in enumerate(latest["sources"], start=1):
            with st.expander(f"Excerpt {i} — Page {src['page']} — Confidence: {src['score']:.2f}"):
                st.write(src["text"])
                st.caption(f"Document: {src['doc_name']} | Chunk ID: {src['chunk_id']}")

# ---------- Query History ----------
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕘 Query History")
    for item in reversed(st.session_state.history[:-1]):
        with st.expander(item["query"]):
            st.write(item["answer"])
