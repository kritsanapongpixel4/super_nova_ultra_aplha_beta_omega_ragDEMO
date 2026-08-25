"""Project configuration — every path and tunable in one place.

The embedding model is the one setting that is *not* edited here any more.
It is selected at runtime, in this order of precedence:

    python build_index.py --model qwen3-0.6b     # command line, wins
    set RAG_EMBED_MODEL=qwen3-0.6b               # environment variable
    src.model_registry.DEFAULT_MODEL             # fallback

Every model gets its own index directory under ``vector_db/<key>/`` and its
own embeddings file, so switching back and forth costs nothing after the
first build and two models can never overwrite each other's index.  See
``src/model_registry.py`` for the list, or run ``python -m src.model_registry``.
"""

import os
from pathlib import Path

from src import model_registry

# --- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"
LOGS_DIR = ROOT_DIR / "logs"
BENCHMARK_DIR = ROOT_DIR / "benchmarks" / "results"

# Supported file types for ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}

# Which category sub-folders of data/ to ingest.
#   ["curriculum"]              → only the หลักสูตร documents (current)
#   ["curriculum", "forms"]     → several categories
#   None                        → everything under data/
SOURCE_CATEGORIES: list[str] | None = ["curriculum"]


def discover_sources() -> list[Path]:
    """Return every supported file in the selected categories.

    Recursive, so files nested inside a category folder are found too, and
    guarded so `import config` still works on a fresh clone where data/ has
    not been created yet.
    """
    if not DATA_DIR.exists():
        return []
    roots = (
        [DATA_DIR]
        if SOURCE_CATEGORIES is None
        else [DATA_DIR / name for name in SOURCE_CATEGORIES]
    )
    return sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


SOURCE_FILES: list[Path] = discover_sources()

# Golden set is optional — not every dataset has one
_golden_candidate = DATA_DIR / "golden_set.json"
GOLDEN_SET_FILE: Path | None = _golden_candidate if _golden_candidate.exists() else None

# Model-independent artefacts: text extraction and chunking depend on the
# documents and the chunking settings, never on which embedder runs next.
EXTRACTED_TEXT_FILE = OUTPUTS_DIR / "extracted_text.json"
CHUNKS_FILE = OUTPUTS_DIR / "chunks.json"
RETRIEVAL_RESULTS_FILE = OUTPUTS_DIR / "retrieval_results.json"
EVAL_RETRIEVAL_FILE = OUTPUTS_DIR / "eval_retrieval.json"
EVAL_GENERATION_FILE = OUTPUTS_DIR / "eval_generation.json"

# BM25 indexes the same chunks for every model, so it is shared too — one
# copy, rebuilt only when chunks.json changes.
BM25_INDEX_FILE = VECTOR_DB_DIR / "bm25_index.pkl"
SPARSE_INDEX_DIR = VECTOR_DB_DIR / "sparse"

# --- Chunking ------------------------------------------------------------
# Tuned for this dataset (Thai university forms + textbooks):
#   - 73.5% of records < 50 tokens, median = 18 tokens
#   - P90 = 128 tokens, P95 = 183 tokens, max = 722 tokens
#   - Small chunk size preserves granularity of short form content
#   - 20% overlap ratio maintains passage continuity
CHUNK_SIZE = 150          # Thai tokens per chunk (measured by PyThaiNLP)
CHUNK_OVERLAP = 30        # Thai tokens shared between neighbouring chunks
MIN_CHUNK_LETTERS = 10    # drop chunks with fewer Thai/Latin letters than this
                          # (page numbers, stray bullets, table furniture).
                          # 20 is too aggressive — it eats course codes.

# --- Embeddings ----------------------------------------------------------
EMBEDDING_BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True         # cosine similarity via inner product
EMBEDDING_DEVICE = os.environ.get("RAG_DEVICE") or None  # None → auto-detect

# Filled in by use_model() immediately below; declared here so the module's
# public surface is visible in one place.
EMBEDDING_KEY: str
EMBEDDING_SPEC: model_registry.EmbeddingSpec
EMBEDDING_MODEL: str
EMBEDDING_DIM: int
MODEL_INDEX_DIR: Path
FAISS_INDEX_FILE: Path
CHUNK_STORE_FILE: Path
INDEX_META_FILE: Path
EMBEDDINGS_FILE: Path


def use_model(key: str) -> model_registry.EmbeddingSpec:
    """Point every model-dependent path at *key* and return its spec.

    Call this before anything imports the paths — ``src/cli.py`` does it
    from the ``--model`` flag, and the module body below does it from
    ``RAG_EMBED_MODEL``.  Rebinding module globals rather than hiding the
    paths behind functions keeps `config.FAISS_INDEX_FILE` working exactly
    as it always did for the twenty-odd places that read it.
    """
    global EMBEDDING_KEY, EMBEDDING_SPEC, EMBEDDING_MODEL, EMBEDDING_DIM
    global MODEL_INDEX_DIR, FAISS_INDEX_FILE, CHUNK_STORE_FILE
    global INDEX_META_FILE, EMBEDDINGS_FILE

    spec = model_registry.resolve(key)
    EMBEDDING_KEY = spec.key
    EMBEDDING_SPEC = spec
    EMBEDDING_MODEL = spec.hf_id
    EMBEDDING_DIM = spec.dim

    MODEL_INDEX_DIR = VECTOR_DB_DIR / spec.slug
    FAISS_INDEX_FILE = MODEL_INDEX_DIR / "document.index"
    CHUNK_STORE_FILE = MODEL_INDEX_DIR / "chunk_store.json"
    INDEX_META_FILE = MODEL_INDEX_DIR / "index_meta.json"
    EMBEDDINGS_FILE = OUTPUTS_DIR / "embeddings" / f"{spec.slug}.npy"
    return spec


use_model(os.environ.get("RAG_EMBED_MODEL") or model_registry.DEFAULT_MODEL)

# --- Retrieval -----------------------------------------------------------
TOP_K = 8                 # chunks handed to the generator.  Raised from 5
                          # because a "which courses cover X" question needs
                          # several course cards, and extra chunks cost almost
                          # nothing against a 1M-token context window.
CANDIDATE_K = 20          # chunks pulled per retriever before fusion/rerank
RRF_K = 60                # Reciprocal Rank Fusion smoothing constant.  Do not
                          # lower it: measured 2026-08-24, dropping to 5 or 1
                          # pushed the right chunk from rank 6 down to 9 and 11
                          # by amplifying BM25's noisy head.
DENSE_WEIGHT = 0.5        # dense vs BM25 weighting when fusing by score

# Which sparse retriever the hybrid pipeline pairs with the dense one.
# benchmarks/bench_retrievers.py compares every option in
# src/sparse_retrievers.py; measured on e5-base / 3,237 chunks (2026-08-25):
#
#                    sparse เดี่ยว   + dense    + dense + ปักหมุด
#   bm25plus            32.8%         89.8%        97.7%   ← ที่ใช้จริง
#   bm25                33.6%         89.1%        96.9%
#   dirichlet-lm        26.6%         92.2%        94.5%
#   tfidf-word          10.2%         79.7%        91.4%
#   tfidf-char           3.9%         62.5%        90.6%
#   bm25l               12.5%         50.0%        86.7%
#
# คอลัมน์ที่ใช้ตัดสินคือคอลัมน์ขวาสุด เพราะ HybridRetriever.retrieve() ตั้ง
# pin_exact_codes=True เป็นค่าเริ่มต้น — นั่นคือของที่รันจริง
#
# dirichlet-lm ชนะตอนไม่ปักหมุด (92.2%) แต่แพ้ตอนปัก (94.5%) เพราะสิ่งที่มัน
# เก่งกว่าคือการหาการ์ดวิชาจากรหัส ซึ่งการปักหมุดทำได้อยู่แล้วและทำได้เป๊ะกว่า
# พอทับซ้อนกัน สิ่งที่เหลือให้วัดคือคำถามที่เหลือ ซึ่ง bm25plus ทำได้ดีกว่า
SPARSE_METHOD = os.environ.get("RAG_SPARSE_METHOD") or "bm25plus"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
USE_RERANKER = False      # It works — it lifted the right chunk from hybrid
                          # rank 6 to 2 — but costs ~291s per query on this
                          # CPU-only box.  Pinning exact course codes
                          # (hybrid_retriever.exact_code_matches) fixes the
                          # same queries outright, in milliseconds.  Turn the
                          # reranker on for the evaluation comparison
                          # (pipeline/complete_retrieval.py --rerank), not for
                          # interactive use.

# --- Generation ----------------------------------------------------------
LLM_PROVIDER = "gemini"           # ผู้ให้บริการที่ src/generator.py เรียกใช้
LLM_MODEL = "gemini-3.5-flash"    # 3.7-flash/flash-latest คืน 503 บ่อย, pro-latest
                                  # ชน 429 ของ free tier, 2.5.* ปิดรับผู้ใช้ใหม่แล้ว
                                  # โควตา free tier แยกรายโมเดล: 3.6-flash หมดก่อน
                                  # 3.5-flash และ 3-flash-preview เป็นตัวสำรอง
                                  # API key อ่านจาก GEMINI_API_KEY ใน .env
# โควตา free tier นับแยกรายโมเดลและรีเซ็ตไม่พร้อมกัน — วันหนึ่งตัวหนึ่งหมด
# อีกตัวยังว่าง  Generator จะไล่ตามลำดับนี้เองเมื่อเจอ 429/503
# แล้วพักโมเดลที่หมดไว้ 10 นาทีก่อนกลับมาลองใหม่
# เรียงตามผลวัดจริง (benchmarks/bench_llm_apis.py, 2026-08-25 สองรอบ รอบละ 6 ครั้ง)
# ไม่ใช่ตามเลขเวอร์ชัน — ลำดับเดิมเรียงกลับด้านกับผลวัด คือเอา 3.7-flash
# ที่ล้มเหลว 8 ใน 12 ครั้งและรอ 75-79 วินาที ไว้เป็นตัวสำรองอันดับแรก
#
#   flash-lite     ผ่าน 12/12  TTFT 1.11-3.37s
#   3-flash-preview ผ่าน 11/12  TTFT 7.26-7.37s
#   3.6-flash      ผ่าน 11/12  TTFT 9.52-30.98s
#   3.7-flash      ผ่าน  4/12  TTFT 75.1-79.5s   ← ท้ายสุด ดีกว่าไม่ตอบเฉยๆ
LLM_FALLBACK_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
)

LLM_MAX_TOKENS = 16000
MEMORY_MAX_TURNS = 6      # conversation turns kept in the prompt

# --- Evaluation ----------------------------------------------------------
EVAL_K_VALUES = (1, 3, 5, 10)


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for directory in (
        DATA_DIR,
        OUTPUTS_DIR,
        OUTPUTS_DIR / "embeddings",
        VECTOR_DB_DIR,
        MODEL_INDEX_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
