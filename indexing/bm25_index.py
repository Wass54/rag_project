"""
bm25_index.py — Keyword-based search using BM25.
"""

import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi
from config import BM25_INDEX_PATH, TOP_K_BM25


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    texts = [chunk["text"] for chunk in chunks]
    tokenized = [tokenize(text) for text in texts]

    index = BM25Okapi(tokenized)

    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "chunks": chunks}, f)

    print(f"BM25 index built: {len(chunks)} chunks saved to {BM25_INDEX_PATH}")
    return index, chunks


def load_bm25_index() -> tuple:
    if not os.path.exists(BM25_INDEX_PATH):
        raise FileNotFoundError("BM25 index not found. Run build_bm25_index() first.")

    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    print(f"BM25 index loaded: {len(data['chunks'])} chunks.")
    return data["index"], data["chunks"]


def bm25_search(query: str, index: BM25Okapi, chunks: list[dict],
                top_k: int = TOP_K_BM25) -> list[dict]:


    tokenized_query = tokenize(query)

    scores = index.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in top_indices:
        if scores[i] == 0:
            continue

        chunk = chunks[i]
        results.append({
            "text": chunk["text"],
            "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
            "score": round(float(scores[i]), 4),
            "metadata": {
                "filename": chunk.get("filename", ""),
                "source": chunk.get("source", ""),
                "parent_id": chunk.get("parent_id", ""),
            }
        })

    return results
