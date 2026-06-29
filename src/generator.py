"""
generator.py
------------
Sends retrieved context + query to the LLM and generates a grounded answer.
Stage: LLM Response Generation
"""

import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

NO_ANSWER_MESSAGE = "The answer is not available in the provided document."

SYSTEM_PROMPT = """You are a strict document Q&A assistant.

Rules you must follow exactly:
1. Answer ONLY using the information in the provided context below.
2. Do NOT use any outside knowledge, assumptions, or information not present in the context.
3. If the context does not contain enough information to answer the question, respond with EXACTLY:
   "The answer is not available in the provided document."
4. Do not guess, infer beyond the text, or fabricate details.
5. Keep answers concise and directly grounded in the supplied excerpts.
"""


class GenerationError(Exception):
    pass


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Excerpt {i} | Page {c['page']} | Score {c['score']}]\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """
    Generate an answer strictly grounded in context_chunks.

    Returns: {"answer": str, "sources": list[dict]}
    If context_chunks is empty, skips the LLM call entirely and returns the
    fallback message (saves an API call and guarantees no hallucination).
    """
    if not context_chunks:
        return {"answer": NO_ANSWER_MESSAGE, "sources": []}

    if not GROQ_API_KEY:
        raise GenerationError("GROQ_API_KEY is not set. Check your .env file.")

    context_block = _build_context_block(context_chunks)
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer based only on the context above."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # deterministic, reduces hallucination
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise GenerationError(f"LLM call failed: {e}")

    return {"answer": answer, "sources": context_chunks}
