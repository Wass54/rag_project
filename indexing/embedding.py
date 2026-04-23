"""
embedding.py — Generate embeddings and store chunks in ChromaDB.

Architecture:
    - CHILD chunks  → embedded and stored in ChromaDB (used for retrieval)
    - PARENT chunks → stored in a Python dict keyed by parent_id
                      (fetched after retrieval to send richer context to the LLM)

Embedding model: all-MiniLM-L6-v2
"""

import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from sentence_transformers import SentenceTransformer
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    DATA_FOLDER,
    BM25_INDEX_PATH,
)


# Load models and clients (done once at import time)
print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


# Collection management

def get_or_create_collection(collection_name: str = CHROMA_COLLECTION):
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity metric
    )
    return collection


def reset_collection(collection_name: str = CHROMA_COLLECTION):
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{collection_name}' created.")
    return collection


# Embedding helpers
def embed_texts(texts: list[str]) -> list[list[float]]:

    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


# Main indexing function
def index_chunks(parents: list[dict], children: list[dict],
                 reset: bool = False) -> dict:
    if reset:
        collection = reset_collection()
    else:
        collection = get_or_create_collection()

    existing_count = collection.count()
    print(f"\nChromaDB collection '{CHROMA_COLLECTION}' has {existing_count} existing documents.")

    print(f"\nEmbedding {len(children)} child chunks...")
    texts = [child["text"] for child in children]
    embeddings = embed_texts(texts)

    ids = [child["chunk_id"] for child in children]

    metadatas = [
        {
            "filename": child["filename"],
            "source": child["source"],
            "parent_id": child.get("parent_id", child["chunk_id"]),
            "chunk_index": child.get("child_index", child.get("chunk_index", 0)),
        }
        for child in children
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Indexed {len(children)} child chunks into ChromaDB.")
    print(f"Total documents in collection: {collection.count()}")

    parent_store = {p["parent_id"]: p for p in parents}


    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    parent_store_path = BM25_INDEX_PATH.replace("bm25_index", "parent_store")

    with open(parent_store_path, "wb") as f:
        pickle.dump(parent_store, f)

    print(f"Parent store saved to: {parent_store_path}")

    return parent_store


# Semantic search
def semantic_search(query: str, collection_name: str = CHROMA_COLLECTION,
                    top_k: int = 10) -> list[dict]:

    collection = get_or_create_collection(collection_name)

    if collection.count() == 0:
        print("Warning: ChromaDB collection is empty. Run indexing first.")
        return []

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "text": results["documents"][0][i],
            "chunk_id": results["ids"][0][i],
            "score": round(1 - results["distances"][0][i], 4),
            "metadata": results["metadatas"][0][i],
        })

    return output


# Load parent store from disk
def load_parent_store() -> dict:
    parent_store_path = BM25_INDEX_PATH.replace("bm25_index", "parent_store")

    if not os.path.exists(parent_store_path):
        print("Warning: parent store not found. Run indexing first.")
        return {}

    with open(parent_store_path, "rb") as f:
        parent_store = pickle.load(f)

    print(f"Parent store loaded: {len(parent_store)} parents.")
    return parent_store

