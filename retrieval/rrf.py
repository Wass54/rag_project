"""
rrf.py — Reciprocal Rank Fusion (RRF)

Merges results from semantic search and BM25 into a single ranked list.
Uses rank position instead of raw scores (which are not comparable).

RRF formula for each chunk:
    rrf_score = sum of 1 / (k + rank) across all result lists
    where k=60 is a constant that reduces the impact of top positions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOP_K_FINAL

K = 60  # standard RRF constant


def reciprocal_rank_fusion(results_lists: list[list[dict]],
                           top_k: int = TOP_K_FINAL) -> list[dict]:

    rrf_scores = {}   # chunk_id -> cumulative RRF score
    chunk_data = {}   # chunk_id -> chunk dict (to retrieve text/metadata later)

    for results in results_lists:
        for rank, chunk in enumerate(results):
            chunk_id = chunk["chunk_id"]

            # Accumulate RRF score — higher rank (lower index) = higher score
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (K + rank + 1)

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = chunk

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for chunk_id in sorted_ids[:top_k]:
        result = chunk_data[chunk_id].copy()
        result["rrf_score"] = round(rrf_scores[chunk_id], 6)
        merged.append(result)

    return merged



# Quick test
if __name__ == "__main__":
    from preprocessing.pdf_loader import load_pdfs_from_folder
    from indexing.chunking import chunk_documents
    from indexing.embedding import index_chunks, semantic_search, load_parent_store
    from indexing.bm25_index import build_bm25_index, bm25_search
    from config import DATA_FOLDER

    docs = load_pdfs_from_folder(DATA_FOLDER)
    parents, children = chunk_documents(docs, strategy="parent_child")

    # Build both indexes
    index_chunks(parents, children, reset=True)
    bm25_index, bm25_chunks = build_bm25_index(children)

    query = "How does the attention mechanism work?"
    print(f"Query: '{query}'\n")

    # Get results from both methods
    sem_results = semantic_search(query, top_k=10)
    bm25_results = bm25_search(query, bm25_index, bm25_chunks, top_k=10)

    print(f"Semantic search: {len(sem_results)} results")
    print(f"BM25 search:     {len(bm25_results)} results")

    # Merge with RRF
    merged = reciprocal_rank_fusion([sem_results, bm25_results], top_k=5)

    print(f"\nAfter RRF fusion — top 5:")
    for i, r in enumerate(merged):
        print(f"[{i+1}] RRF score: {r['rrf_score']} | {r['metadata']['filename']}")
        print(f"     {r['text'][:150]}...")