"""
llm_answer.py — Build context prompt and generate answer via Groq API.

The LLM only sees the reranked chunks as context.
It is instructed to answer strictly from those sources.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)


def build_context(chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks):
        filename = chunk["metadata"].get("filename", "unknown")
        ce_score = chunk.get("ce_score", "?")
        text = chunk["text"]

        context_parts.append(
            f"[Source {i+1} — {filename} | relevance: {ce_score}]\n{text}"
        )

    return "\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> list[dict]:
    system_message = """You are a precise and helpful assistant specialized in answering questions about technical documents.

Your rules:
- Answer ONLY based on the provided source passages.
- If the answer is not found in the sources, say: "I could not find this information in the provided documents."
- Always cite which source(s) you used (e.g. "According to Source 1...").
- Be concise and structured in your answers.
- Do not use any external knowledge beyond what is in the sources."""

    user_message = f"""Here are the relevant passages from the documents:

{context}

---

Question: {query}

Answer based strictly on the passages above:"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant passages found in the documents for this question."

    context = build_context(chunks)
    messages = build_prompt(query, context)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,  # low temperature = more factual, less creative
        max_tokens=1024,
    )

    return response.choices[0].message.content
