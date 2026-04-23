"""
streamlit_app.py — Web interface for the RAG pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from preprocessing.pdf_loader import load_pdfs_from_folder
from indexing.chunking import chunk_documents
from indexing.embedding import index_chunks, semantic_search, load_parent_store
from indexing.bm25_index import build_bm25_index, bm25_search
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import rerank
from generation.llm_answer import generate_answer
from config import DATA_FOLDER


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Pipeline",
    layout="wide"
)

st.title("Hybrid RAG Pipeline")
st.caption("Semantic search + BM25 + Cross-Encoder reranking + Groq LLM")


@st.cache_resource(show_spinner="Building index...")
def load_indexes():
    docs = load_pdfs_from_folder(DATA_FOLDER)
    parents, children = chunk_documents(docs, strategy="parent_child")
    index_chunks(parents, children, reset=False)
    bm25_index, bm25_chunks = build_bm25_index(children)
    parent_store = load_parent_store()
    return bm25_index, bm25_chunks, parent_store, len(children), [d["filename"] for d in docs]

bm25_index, bm25_chunks, parent_store, n_chunks, filenames = load_indexes()



with st.sidebar:
    st.header("Indexed documents")
    for f in filenames:
        st.markdown(f"- `{f}`")
    st.metric("Total chunks", n_chunks)
    st.divider()
    st.header("Settings")
    top_k = st.slider("Chunks sent to LLM", min_value=1, max_value=10, value=5)
    show_sources = st.toggle("Show source chunks", value=True)


query = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. How does the attention mechanism work?"
)

if query:
    with st.spinner("Searching and generating answer..."):

        # Retrieve
        sem_results  = semantic_search(query, top_k=10)
        bm25_results = bm25_search(query, bm25_index, bm25_chunks, top_k=10)
        fused        = reciprocal_rank_fusion([sem_results, bm25_results], top_k=10)
        reranked     = rerank(query, fused, top_k=top_k)

        # Generate
        answer = generate_answer(query, reranked)

    # Display answer
    st.subheader("Answer")
    st.markdown(answer)

    # Display source chunks
    if show_sources and reranked:
        st.divider()
        st.subheader("Source chunks used")
        for i, chunk in enumerate(reranked):
            with st.expander(
                    f"Source {i+1} — {chunk['metadata']['filename']} "
                    f"(CE score: {chunk.get('ce_score', '?')})"
            ):
                st.text(chunk["text"])