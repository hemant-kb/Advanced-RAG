"""
Central configuration for the RAG chatbot.

ALL model names, Qdrant settings, chunking parameters, and pipeline limits
live here. To change a model or tune retrieval, edit only this file.
"""
import logging
import os
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Data directories ─────────────────────────────────────────────
# All runtime data lives under data/ and logs/ at the project root.
# Paths are relative to the working directory (project root).
DATA_DIR = "data"
LOG_DIR  = "logs"

Path(DATA_DIR).mkdir(exist_ok=True)
Path(LOG_DIR).mkdir(exist_ok=True)

# ── Request logger (one JSON line per chat request) ──────────────
# Writes to logs/requests.jsonl (rotating, max 10 MB × 3 files)
# AND to stdout so uvicorn captures it too.
_req_logger = logging.getLogger("rag.requests")
if not _req_logger.handlers:
    _req_logger.setLevel(logging.INFO)
    _req_logger.propagate = False

    _file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "requests.jsonl"),
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter("%(message)s"))
    _req_logger.addHandler(_file_handler)

    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(logging.Formatter("%(message)s"))
    _req_logger.addHandler(_stream_handler)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Suppress httpx INFO logs (Qdrant health-check GETs generate noise at INFO level)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress PyMuPDF suggestion about the optional pymupdf_layout package
warnings.filterwarnings("ignore", message=".*pymupdf_layout.*")

# ── Pipeline mode ───────────────────────────────────────────────
# Controls table processing strategy. Set by user on UI; stored per session.
# Valid values:
#   low_cost     — markdown tables only
#   hybrid       — rule-based table routing (simple→markdown, complex→VLM)
#   high_quality — all tables via VLM
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "hybrid")

# ── Models ──────────────────────────────────────────────────────
CHAT_MODEL      = "gpt-5-nano"          # RAG text generation + answer synthesis
VISION_MODEL    = "gpt-5-nano"          # image + table VLM captioning (all modes)

# ── LLM pricing ($ per 1M tokens) — rates for CHAT_MODEL above ──
# Used only for the per-query cost estimate shown in the UI pipeline summary.
# Update these whenever CHAT_MODEL changes.
CHAT_MODEL_PRICE_INPUT  = 0.10    # $ per 1M input tokens
CHAT_MODEL_PRICE_OUTPUT = 0.40    # $ per 1M output tokens
CHAT_MODEL_PRICE_CACHED = 0.025   # $ per 1M cached input tokens

# ── OpenAI embedding (API) ──────────────────────────────────────
# text-embedding-3-small: 1536-dim, 8191-token context, ~$0.02 per 1M tokens.
# BM25 sparse vectors come from FastEmbed (Qdrant/bm25) independently, so the
# hybrid dense+sparse pipeline is unaffected by using an API dense embedder.
EMBEDDING_MODEL = "text-embedding-3-small"       # 1536-dim, OpenAI API

# ── Reranker — Cohere API (replaces local BGE, ~300ms vs ~25s) ──
# rerank-v4.0-fast: low-latency, strong quality, ~$0.0001 per 20 pairs.
# BGE reranker was 23-27s per query; Cohere is ~200-400ms.
COHERE_API_KEY      = os.environ.get("COHERE_API_KEY")
COHERE_RERANK_MODEL = "rerank-v4.0-fast"

# ── Qdrant ──────────────────────────────────────────────────────
QDRANT_URL               = os.environ.get("QDRANT_URL")
QDRANT_API_KEY           = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION_PREFIX = "session"  # collections named: session_{session_id}
QDRANT_VECTOR_SIZE       = 1536       # text-embedding-3-small output dim
QDRANT_SEARCH_LIMIT      = 5
QDRANT_TIMEOUT           = 120
QDRANT_SPARSE_VECTOR_NAME = "sparse"   # name of the sparse vector field in Qdrant
HYBRID_PREFETCH_LIMIT     = 10         # candidates fetched before reranking (was 20)
RERANK_TOP_N              = 5          # final chunks after reranking

# ── Document Pipeline ───────────────────────────────────────────
# Parent-document retrieval: child chunks are searched, parent chunks sent to LLM
# Sizes are in tiktoken (cl100k_base) tokens — same tokenizer family as
# text-embedding-3-small, whose hard limit is 8191 tokens.
CHILD_CHUNK_SIZE       = 256    # tokens — embedded + searched
CHILD_CHUNK_OVERLAP    = 32
PARENT_CHUNK_SIZE      = 512    # tokens — passed to LLM as context
PARENT_CHUNK_OVERLAP   = 50
MIN_IMAGE_AREA         = 10_000       # ignore tiny decorative images (px²)
IMAGE_CAPTION_PROMPT   = (
    "Describe this image/chart for document retrieval. Be specific about data, "
    "trends, labels, axes, values, and any text visible in the image. "
    "Write a dense, factual description optimized for semantic search."
)

PIPELINE_MAX_WORKERS  = 4       # parallel page processing threads
TABLE_MIN_ROWS        = 2       # ignore tables with fewer rows
TABLE_MIN_COLS        = 2       # ignore tables with fewer cols

# Hybrid table complexity thresholds — if any is exceeded, route to VLM
TABLE_VLM_MAX_ROWS        = 30    # more rows → VLM
TABLE_VLM_MAX_COLS        = 8     # more cols → VLM
TABLE_VLM_EMPTY_RATIO     = 0.3   # fraction of empty cells → likely merged → VLM
TABLE_VLM_HEADER_DEPTH    = 1     # multi-level headers → VLM

TABLE_VLM_PROMPT = (
    "This is a table from a document. Extract all data accurately in markdown format. "
    "Preserve all headers, values, units, and structure. "
    "If cells are merged, represent them clearly. "
    "Output only the markdown table, nothing else."
)
HEADER_FOOTER_MARGIN  = 0.08    # top/bottom page fraction treated as header/footer

CHART_CAPTION_PROMPT  = (
    "This is a chart or graph from a document. Describe it precisely for document retrieval: "
    "identify the chart type (bar, line, pie, scatter, etc.), all axis labels and units, "
    "the time range or categories shown, key data points, trends, and any title or legend text. "
    "Write a dense factual description optimized for semantic search."
)

# ── RAG Graph ───────────────────────────────────────────────────
MAX_REWRITE_COUNT      = 1

# Minimum Cohere rerank score for the top chunk to be considered relevant.
# Below this threshold the query is rewritten and retrieval retried once.
# Cohere rerank-v4.0-fast scores range ~0.0–1.0; 0.45 catches weak matches
# while avoiding unnecessary rewrites on confidently-retrieved content.
RELEVANCY_SCORE_THRESHOLD = 0.45

# Token the chat_llm emits when the caption is insufficient and
# the original image must be loaded for a proper answer.
NEEDS_VISUAL_TOKEN = "[NEEDS_VISUAL]"

# Static system prompt — sent as the "system" role so OpenAI can cache it.
# Kept stable across requests; only the user message (context + question) varies.
CAPTION_ANSWER_SYSTEM = (
    "You are a precise document assistant. Answer questions using ONLY the provided context.\n"
    "Do not use external knowledge or invent information.\n"
    "The context may include image/chart captions (text descriptions of visuals).\n\n"
    "Formatting rules:\n"
    "- Use **bold** for key numbers, limits, dates, and named entities.\n"
    "- Use bullet points when listing multiple facts, conditions, or items.\n"
    "- For a single-fact answer, one or two sentences is enough — no forced structure.\n"
    "- Never use prose paragraphs when a bullet list would be clearer.\n"
    "- Maximum 150 words unless the question genuinely requires more.\n\n"
    "Content rules:\n"
    "1. If the context contains a clear answer, respond directly.\n"
    "2. If the context is partially relevant, answer only what is supported and note what is missing.\n"
    "3. If the context contains NO relevant information, respond with exactly:\n"
    "   \"The uploaded document does not contain information about this topic.\"\n"
    "4. If a visual is referenced and the caption is clearly insufficient for the question, "
    "output ONLY the token: [NEEDS_VISUAL]\n"
    "5. When the context states different rules for different conditions (e.g. first child vs second, "
    "under 7 days vs 7+ days), state EACH rule separately and accurately. Never merge or conflate them."
)

# User message template — only the context and question vary per request.
CAPTION_ANSWER_PROMPT = "Context:\n{context}\n\nQuestion: {query}"

SUMMARY_CHUNK_PROMPT = (
    "Summarise this document section in 3-5 bullet points. "
    "Each bullet: one sentence, specific — include numbers, names, dates, limits. "
    "No padding, no commentary. Output bullets only.\n\n"
    "Text:\n{text}"
)

SUMMARY_FINAL_PROMPT = (
    "Write a document summary as structured bullet points. "
    "Use these sections (omit any with no relevant content):\n"
    "• What this document is about (1-2 bullets)\n"
    "• Key policies / rules / findings (3-6 bullets, include numbers)\n"
    "• Important limits, dates, or conditions (2-4 bullets)\n"
    "• Any recommendations or actions required (1-3 bullets)\n\n"
    "Rules: one fact per bullet, be specific, max 20 bullets total, no prose paragraphs.\n\n"
    "Section summaries:\n{summaries}"
)

# ── LLM output limit ────────────────────────────────────────────
MAX_TOKENS        = 4000  # max completion tokens — reasoning models (gpt-5-nano) use tokens for internal CoT before outputting; 2000 was sometimes exhausted by CoT alone on complex multi-condition questions
MAX_TOKENS_VISION = 1500  # vision LLM may need more for image descriptions

# ── External APIs ───────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ── Groq (used for document summarisation at ingest + evaluation) ──
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY")
GROQ_BASE_URL     = "https://api.groq.com/openai/v1"
GROQ_SUMMARY_MODEL = "openai/gpt-oss-120b"
SUMMARY_BATCH_TOKENS = 4000   # ~chars per batch = tokens * 4
SUMMARY_BATCH_CHARS  = 16000  # 4000 tokens * 4 chars/token

# ── LangSmith tracing ────────────────────────────────────────────
# .env uses LANGSMITH_* names (the modern LangSmith SDK convention).
# We also set the LANGCHAIN_* aliases so langchain + langsmith both pick them up.
LANGSMITH_API_KEY  = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT  = os.environ.get("LANGSMITH_PROJECT", "production-ai-assistant")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_TRACING  = os.environ.get("LANGSMITH_TRACING", "true")

# Mirror into the LANGCHAIN_* env vars that langchain-core reads internally
os.environ.setdefault("LANGCHAIN_TRACING_V2",  LANGSMITH_TRACING)
os.environ.setdefault("LANGSMITH_PROJECT",      LANGSMITH_PROJECT)
os.environ.setdefault("LANGSMITH_ENDPOINT",     LANGSMITH_ENDPOINT)
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY",  LANGSMITH_API_KEY)

# ── Persistence ─────────────────────────────────────────────────
CHECKPOINT_DB       = os.path.join(DATA_DIR, "checkpoints.db")
SESSIONS_DB         = os.path.join(DATA_DIR, "sessions.db")
UPLOAD_DIR          = os.path.join(DATA_DIR, "uploads")
IMAGE_STORE_DIR     = os.path.join(DATA_DIR, "images")   # saved PNGs: images/{session_id}/{page}_{xref}.png
EMBEDDING_CACHE_DIR = os.path.join(DATA_DIR, "embedding_cache")

# ── Evaluation ──────────────────────────────────────────────────
EVAL_MODEL         = "gpt-4.1-mini"  # DeepEval judge LLM — better judgment than nano, still in GPTModel allowlist
EVAL_CONCURRENCY   = 3   # concurrent metric evaluations
EVAL_THROTTLE      = 1   # seconds between metric calls

# Per-metric pass thresholds — used by DeepEval and compare.py.
# Contextual Relevancy is intentionally lower: policy-doc chunks contain
# multiple leave types per chunk, so irrelevant sentences are always present.
EVAL_THRESHOLDS = {
    "contextual_precision":   0.80,
    "contextual_recall":      0.70,
    "contextual_relevancy":   0.40,  # lower — policy chunks are multi-topic by nature
    "faithfulness":           0.90,
    "answer_relevancy":       0.80,
    "correctness":            0.85,
}
# Backward-compatible single threshold used by metrics that don't have a named entry above.
EVAL_THRESHOLD     = 0.7

EVAL_DIR           = os.path.join("backend", "evaluate")
GOLDENS_DIR        = os.path.join(EVAL_DIR, "goldens")
REPORTS_DIR        = os.path.join(EVAL_DIR, "reports")
GOLDENS_FILE       = os.path.join(GOLDENS_DIR, "axtria_leave_policy.json")
EVAL_RESULTS_FILE  = os.path.join(REPORTS_DIR, "latest.json")

# ── Collection helper ───────────────────────────────────────────
def get_collection_name(session_id: str) -> str:
    """Qdrant collection name from a session id (sanitised for Qdrant)."""
    safe = session_id.replace("-", "_").replace(" ", "_")
    return f"{QDRANT_COLLECTION_PREFIX}_{safe}"
