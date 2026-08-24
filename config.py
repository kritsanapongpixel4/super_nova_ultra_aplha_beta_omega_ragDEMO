"""Project configuration — every path and tunable in one place."""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"

# Supported file types for ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}

# Auto-discover all supported files in data/
SOURCE_FILES: list[Path] = sorted(
    p for p in DATA_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
)

# Golden set is optional — not every dataset has one
_golden_candidate = DATA_DIR / "golden_set.json"
GOLDEN_SET_FILE: Path | None = _golden_candidate if _golden_candidate.exists() else None

EXTRACTED_TEXT_FILE = OUTPUTS_DIR / "extracted_text.json"
CHUNKS_FILE = OUTPUTS_DIR / "chunks.json"
EMBEDDINGS_FILE = OUTPUTS_DIR / "embeddings.npy"
RETRIEVAL_RESULTS_FILE = OUTPUTS_DIR / "retrieval_results.json"
EVAL_RETRIEVAL_FILE = OUTPUTS_DIR / "eval_retrieval.json"
EVAL_GENERATION_FILE = OUTPUTS_DIR / "eval_generation.json"

FAISS_INDEX_FILE = VECTOR_DB_DIR / "document.index"
BM25_INDEX_FILE = VECTOR_DB_DIR / "bm25_index.pkl"
CHUNK_STORE_FILE = VECTOR_DB_DIR / "chunk_store.json"
INDEX_META_FILE = VECTOR_DB_DIR / "index_meta.json"

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 500          # Thai tokens per chunk (measured by PyThaiNLP)
CHUNK_OVERLAP = 50        # Thai tokens shared between neighbouring chunks

# --- Embeddings ----------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-m3"     # multilingual, works well with Thai
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True         # cosine similarity via inner product

# --- Retrieval -----------------------------------------------------------
TOP_K = 5                 # chunks handed to the generator
CANDIDATE_K = 20          # chunks pulled per retriever before fusion/rerank
RRF_K = 60                # Reciprocal Rank Fusion smoothing constant
DENSE_WEIGHT = 0.5        # dense vs BM25 weighting when fusing by score
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
USE_RERANKER = True

# --- Generation ----------------------------------------------------------
LLM_MODEL = "claude-opus-5"
LLM_MAX_TOKENS = 16000
MEMORY_MAX_TURNS = 6      # conversation turns kept in the prompt

# --- Evaluation ----------------------------------------------------------
EVAL_K_VALUES = (1, 3, 5, 10)


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for directory in (DATA_DIR, OUTPUTS_DIR, VECTOR_DB_DIR):
        directory.mkdir(parents=True, exist_ok=True)
