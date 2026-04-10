import os
from dotenv import load_dotenv

load_dotenv()

# LLM SETTINGS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"

# EMBEDDING MODEL
# Lightweight model: 384 dimensions
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# VECTOR STORE — ChromaDB
CHROMA_DB_PATH = "./data/chroma_db"
CHROMA_COLLECTION = "rag_docs"

# CHUNKING
CHUNK_SIZE = 512        # target chunk size in tokens
CHUNK_OVERLAP = 64      # overlap between consecutive chunks to avoid losing context at boundaries

# Parent-Child strategy: child chunks are used for retrieval (precise),
# parent chunks are sent to the LLM (more context)
PARENT_CHUNK_SIZE = 1024

# RETRIEVAL
TOP_K_SEMANTIC = 10     # number of results from semantic search
TOP_K_BM25 = 10         # number of results from BM25 keyword search
TOP_K_FINAL = 5         # number of chunks kept after reranking

# RERANKER
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunks with a reranker score below this threshold are considered out-of-scope
CE_RELEVANCE_THRESHOLD = 0.3

# PATHS
DATA_FOLDER = "./data/pdfs"
BM25_INDEX_PATH = "./data/bm25_index.pkl"