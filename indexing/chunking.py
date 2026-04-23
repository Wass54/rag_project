"""
chunking.py — Split documents into chunks for indexing.

Two strategies are implemented:

1. Simple chunking
2. Parent-Child chunking
"""

import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE

def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a long text into overlapping chunks of roughly `chunk_size` words.

    We split on words (not characters or tokens) for simplicity.
    In production you would use a proper tokenizer (tiktoken, HuggingFace)
    to count tokens precisely — but word count is a good approximation
    and avoids adding a heavy dependency at this stage.

    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        # Grab the slice of words for this chunk
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(chunk_text)

        # Move forward by (chunk_size - overlap) so the next chunk
        # re-uses the last `overlap` words of the current one
        step = chunk_size - overlap
        start += step

        # Safety guard: if overlap >= chunk_size we would loop forever
        if step <= 0:
            break

    return chunks


# Simple chunking
def chunk_document(document: dict, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a single document into simple overlapping chunks.

    Each chunk dict contains:
    - text:       the chunk content
    - chunk_id:   unique identifier  "<filename>_chunk_<index>"
    - source:     original file path
    - filename:   original filename
    - chunk_index: position of this chunk in the document
    """
    text = document["text"]
    filename = document["filename"]
    source = document["source"]

    raw_chunks = split_text_into_chunks(text, chunk_size, overlap)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "text": chunk_text,
            "chunk_id": f"{filename}_chunk_{i}",
            "source": source,
            "filename": filename,
            "chunk_index": i,
        })

    return chunks


# Parent-Child chunking
def chunk_document_parent_child(document: dict,
                                child_size: int = CHUNK_SIZE,
                                parent_size: int = PARENT_CHUNK_SIZE,
                                overlap: int = CHUNK_OVERLAP) -> tuple[list[dict], list[dict]]:
    """
    Split a document into parent chunks and child chunks.

    Workflow:
        1. Split the document into large PARENT chunks (e.g. 1024 words).
        2. For each parent, split it further into small CHILD chunks (e.g. 512 words).
        3. Each child stores a reference to its parent via `parent_id`.

    At retrieval time:
        - We search over CHILD chunks (small → more precise embedding match).
        - We then fetch the full PARENT chunk to send to the LLM (more context).
    """
    text = document["text"]
    filename = document["filename"]
    source = document["source"]

    raw_parents = split_text_into_chunks(text, parent_size, overlap=0)

    parents = []
    children = []

    for p_idx, parent_text in enumerate(raw_parents):
        parent_id = f"{filename}_parent_{p_idx}"

        parent_chunk = {
            "text": parent_text,
            "parent_id": parent_id,
            "source": source,
            "filename": filename,
            "parent_index": p_idx,
        }
        parents.append(parent_chunk)

        raw_children = split_text_into_chunks(parent_text, child_size, overlap)

        for c_idx, child_text in enumerate(raw_children):
            child_chunk = {
                "text": child_text,
                "chunk_id": f"{filename}_parent_{p_idx}_child_{c_idx}",
                "parent_id": parent_id,
                "source": source,
                "filename": filename,
                "parent_index": p_idx,
                "child_index": c_idx,
            }
            children.append(child_chunk)

    return parents, children


# Batch processing — handle a list of documents
def chunk_documents(documents: list[dict],
                    strategy: str = "parent_child") -> tuple[list[dict], list[dict]]:

    all_parents = []
    all_children = []

    for doc in documents:
        if strategy == "parent_child":
            parents, children = chunk_document_parent_child(doc)
            all_parents.extend(parents)
            all_children.extend(children)
        else:
            chunks = chunk_document(doc)
            all_children.extend(chunks)

    print(f"\nChunking complete ({strategy} strategy):")
    if strategy == "parent_child":
        print(f"  - {len(all_parents)} parent chunks")
        print(f"  - {len(all_children)} child chunks")
    else:
        print(f"  - {len(all_children)} chunks")

    return all_parents, all_children
