"""
reranker.py — Rerank chunks using a Cross-Encoder model.

The Cross-Encoder reads (question + chunk) together and outputs
a relevance score. Much more precise than embedding similarity
but slower only used on the top 5-10 chunks from RRF.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, CE_RELEVANCE_THRESHOLD, TOP_K_FINAL


print(f"Loading reranker: {RERANKER_MODEL}")
reranker = CrossEncoder(RERANKER_MODEL)


def rerank(query: str, chunks: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    """
    Rerank a list of chunks using the Cross-Encoder.

    For each chunk, the model reads the pair (query, chunk text) together
    and outputs a single relevance score.

    Chunks below CE_RELEVANCE_THRESHOLD are filtered out they are
    considered out-of-scope and we don't want to send them to the LLM.
    We only take top 5
    """
    if not chunks:
        return []

    pairs = [(query, chunk["text"]) for chunk in chunks]

    scores = reranker.predict(pairs)
    for i, chunk in enumerate(chunks):
        chunk["ce_score"] = round(float(scores[i]), 4)

    ranked = sorted(chunks, key=lambda x: x["ce_score"], reverse=True)

    relevant = [c for c in ranked if c["ce_score"] >= CE_RELEVANCE_THRESHOLD]

    if not relevant:
        print("Warning: no chunks passed the relevance threshold.")
        return ranked[:1]

    return relevant[:top_k]
