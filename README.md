# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) system that answers questions strictly
from uploaded PDF content, using Pinecone as the vector database. Built to
prevent hallucination and provide traceable, page-level source attribution.

## Architecture

```
PDF Upload → Text Extraction → Text Chunking → Embedding Generation
→ Pinecone Vector Indexing → Semantic Retrieval → LLM Response Generation
→ Answer with Source Reference
```

## Project Structure

```
rag-pdf-chatbot/
├── app.py                     # Streamlit UI — entrypoint
├── src/
│   ├── pdf_loader.py            # PDF validation + text extraction + cleaning
│   ├── chunker.py                # LangChain RecursiveCharacterTextSplitter chunking
│   ├── embedder.py                # Sentence-Transformer embeddings (LangChain HuggingFaceEmbeddings, cached)
│   ├── pinecone_client.py         # Index creation, upsert, query, namespaces (cached for performance)
│   ├── retriever.py                # Top-k retrieval + similarity threshold
│   ├── generator.py                # Groq LLM call + hallucination guardrails
│   └── utils.py                     # Namespace naming, query logging
├── logs/
│   └── query_log.csv             # Logged queries (created at runtime)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone and create a virtual environment**
   ```bash
   git clone <your-repo-url>
   cd rag-pdf-chatbot
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Then fill in `.env` with:
   - `PINECONE_API_KEY` — from your [Pinecone console](https://app.pinecone.io)
   - `GROQ_API_KEY` — from [console.groq.com](https://console.groq.com)

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

   > **Known issue:** if you see `Client.init() got an unexpected keyword argument 'proxies'`,
   > your installed `groq` package is incompatible with the `httpx` version pip resolved.
   > Fix with `pip install --upgrade groq` and restart Streamlit.

## How It Works

The interface is a simple form-based layout: settings live in an always-visible sidebar, the main page has a numbered flow (1. Upload a PDF → 2. Ask a Question → 3. Answer), with a Source Attribution section showing page, similarity score, and excerpt for each chunk used, followed by a Query History section listing previous questions and answers from the session.

1. **Upload** — One or more PDFs (≤20MB each) are uploaded via the "1. Upload a PDF" section on the main page.
2. **Extraction** — `pdf_loader.py` extracts text per page and cleans formatting artifacts (hyphenation, extra whitespace).
3. **Chunking** — `chunker.py` uses **LangChain's `RecursiveCharacterTextSplitter`** to split each page into overlapping chunks, preferring paragraph/sentence boundaries over blind character cuts (size/overlap adjustable from the sidebar, in characters).
4. **Embedding** — `embedder.py` wraps `sentence-transformers/all-MiniLM-L6-v2` in **LangChain's `HuggingFaceEmbeddings`**, exposing the standard `.embed_documents` / `.embed_query` interface used throughout the pipeline.
5. **Indexing** — `pinecone_client.py` creates a Pinecone serverless index directly via the Pinecone SDK (explicit index creation/readiness check), then stores/retrieves vectors through **LangChain's `PineconeVectorStore`**, one namespace per document, with metadata (`doc_name`, `page`, `chunk_id`) attached to each `Document`.
6. **Retrieval** — `retriever.py` embeds the user's query, then calls `pinecone_client.query_index()`, which queries the vector store via `similarity_search_by_vector_with_score` and filters results by an adjustable similarity threshold.
7. **Generation** — `generator.py` sends only the retrieved chunks to the LLM (Groq, `openai/gpt-oss-20b`) with a strict system prompt. If no chunks clear the threshold, the LLM is never called — the app returns the fallback message directly, guaranteeing no hallucinated answer.
8. **Source Attribution** — The "3. Answer" section is followed by an expandable "Source Attribution" entry for each chunk used, showing page number, excerpt text, similarity (confidence) score, and chunk ID. Past questions and answers remain visible below in the "Query History" section for the rest of the session.

### Performance: caching

The embedding model, Pinecone client, index handle, and vector store are all wrapped in `st.cache_resource` (in `embedder.py` and `pinecone_client.py`). Without this, Streamlit's rerun-on-every-interaction model would re-check the Pinecone index's existence and reload the embedding model on every single question, adding unnecessary network round-trips and load time to each answer. With caching, that setup work happens once per server session — only the first upload/question is slow; subsequent questions skip straight to embedding the query and calling Pinecone/Groq.

### Why LangChain is used the way it is

LangChain is used for the two stages where it adds real value without hiding what's happening: **text splitting** (`RecursiveCharacterTextSplitter` is genuinely better than a naive word-window splitter) and the **vector store interface** (`PineconeVectorStore` gives a standard `Document`-based API for upsert/query). Index creation and namespace/readiness management are still done directly through the Pinecone SDK rather than left implicit inside LangChain, since the assignment specifically asks for visible, demonstrable control over index creation, namespaces, upserting, and metadata.

## Implemented Enhancements (intermediate-level, 6 of 7 required minimum 3)

- ✅ Multi-document support (separate Pinecone namespace per PDF, query across all or filter to one)
- ✅ Query history (session-level, shown in UI)
- ✅ Adjustable chunk size / overlap from the sidebar
- ✅ Adjustable top-k retrieval from the sidebar
- ✅ Metadata filtering by page number
- ✅ Confidence scoring displayed per source excerpt
- ✅ Logging of all user queries to `logs/query_log.csv`

## Hallucination Prevention Strategy

- Retrieval results below the similarity threshold are discarded before reaching the LLM.
- If zero chunks pass the threshold, the LLM call is skipped entirely and the fixed fallback message is returned.
- The system prompt explicitly forbids outside knowledge and mandates the exact fallback phrase when context is insufficient.
- `temperature=0.0` is used to minimize generative drift from the supplied context.

**Empirical threshold tuning:** during testing, the default similarity threshold (0.3) was too strict — `all-MiniLM-L6-v2` cosine scores for genuinely correct matches on short factual Q&A often land in the 0.15–0.3 range rather than near 1.0, so valid answers were incorrectly rejected as "not available in the document." Lowering the threshold (e.g. to ~0.15–0.2) resolved this. This is a real precision/recall tradeoff worth documenting: too high a threshold causes false negatives (rejecting valid answers), too low risks false positives (letting irrelevant chunks reach the LLM).

## Notes

- Embeddings are generated locally (no API cost) using Sentence-Transformers.
- Pinecone uses serverless indexes (AWS, configurable region) with cosine similarity.
- Encrypted or scanned/image-only PDFs are rejected with a clear error message (no OCR is performed in this version).
