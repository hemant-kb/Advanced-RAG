# CLAUDE.md — Multimodal RAG Chatbot

Project reference for Claude Code. Detailed architecture + diagrams: `docs/ARCHITECTURE.md` and `docs/images/*.svg`. Keep both this file and the docs in sync when the pipeline or graph changes.

## What this is

A **multimodal RAG chatbot** — PDF Q&A over text, tables, images and charts (no OCR, digital PDFs only). Single mode: every query goes through the RAG subgraph; no general chat, no direct-answer bypass. Summary queries take a fast path that reads a pre-generated summary chunk.

**Stack:** FastAPI + LangGraph backend · React + TypeScript (Vite) frontend · Qdrant Cloud (hybrid dense + BM25) · OpenAI `gpt-5-nano` (chat + vision) + `text-embedding-3-small` · Cohere `rerank-v4.0-fast` · Groq `gpt-oss-120b` (ingest-time summaries) · LangSmith tracing · DeepEval evaluation.

## How to run

```powershell
# Backend — MUST run from the project root (absolute `backend.*` imports everywhere;
# running from inside backend/ gives ModuleNotFoundError: No module named 'backend')
cd "C:\Users\A6759\OneDrive - Axtria\Downloads\Projects\RAG\project"
uvicorn backend.api:app --reload          # http://localhost:8000

# Frontend (Vite proxies /api → localhost:8000)
cd frontend; npm run dev                  # http://localhost:5173

# Evaluation / regression gate
python -m backend.evaluate.evaluate <pdf> backend/evaluate/goldens/axtria_leave_policy.json [max_cases]
python -m backend.evaluate.compare backend/evaluate/reports/baseline.json backend/evaluate/reports/latest.json
```

## Layout

```
backend/
  config.py            # SINGLE SOURCE OF TRUTH — all models, prices, thresholds, chunk sizes, prompts, paths
  models.py            # Pydantic schemas (API requests + SessionInfo/UploadStatus)
  graph.py             # master LangGraph: one node = compiled RAG subgraph
  api.py               # thin app assembler: lifespan (app.state.graph), CORS, routers, /health
  session_store.py     # SQLite session registry (per-call WAL connections)
  routes/
    sessions.py        # CRUD, auto-name (gpt-4.1-nano), history, cascading delete
    upload.py          # PDF upload + SSE ingestion progress; in-memory _upload_status
    chat.py            # SSE chat streaming, PipelineMetrics, NODE_LABELS, context-limit guard
    evaluation.py      # POST /evaluate → DeepEval subprocess
  rag/
    document_pipeline.py  # ingest_pdf: _extract_pages (phase 1) + _build_text_chunks (phase 2) + Groq summary
    rag_graph.py          # RAG subgraph, session-keyed _emit(), badge dicts, two-phase answer
    vector_store.py       # Qdrant hybrid search, Cohere rerank, structural promotion
    guardrails.py         # prompt-injection patterns (guardrails 2 & 3 live at their point of use)
    audit.py              # IngestionAudit + QueryAudit → data/runs/{session_id}/
  evaluate/
    evaluate.py        # 6 LLM metrics (gpt-4.1-mini judge) + 4 perf metrics (MeasuredValueMetric)
    compare.py         # baseline diff; Δ ≥ 0.05 = regression, exit 1
frontend/src/
  App.tsx              # sessions, theme, turnMetaMapRef (survives ChatWindow remounts)
  hooks/useStream.ts   # SSE reader; ThinkingStep / Badge / TurnMeta / PipelineSummary types
  components/          # ChatWindow, Sidebar, MessageBubble, ThinkingBlock, PipelineSummary,
                       # ConfirmDialog, icons.tsx (all shared inline SVGs)
data/                  # runtime, gitignored: sessions.db, checkpoints.db, uploads/, images/, runs/
logs/requests.jsonl    # one JSON line per chat request (rotating)
```

## Architecture in one pass

- **Ingestion** (`ingest_pdf`): PyMuPDF4LLM markdown → phase 1 per-page in 4 parallel threads (tables: markdown or VLM by `PIPELINE_MODE`; images: filter → PNG to disk → VLM caption as `page_content`) → phase 2 full-doc heading-aware split → parent 512 / child 256 tiktoken chunks (never cross sections) → dense (contextual `Title/H1-H3` prefix) + BM25 sparse upsert → Groq map-reduce summary stored as `type="document_summary"` chunk.
- **Query graph** (`rag_graph.py`): `entry` (injection guardrail) → summary fast-path OR `retrieval` (hybrid RRF prefetch 10 → Cohere rerank top 5 → structural promotion) → `relevancy_check` (top rerank score ≥ 0.45, no LLM) → one `query_rewrite` retry on miss → `generate_answer` (Phase 1 captions/text; Phase 2 vision only on `[NEEDS_VISUAL]`).
- **Streaming**: nodes call `_emit(session_id, step)`; `routes/chat.py` drains the queue between `astream_events` events into `thinking_step` SSE events; `PipelineMetrics` folds badge dicts into the end-of-turn `pipeline_summary`.
- **Evaluation**: goldens → ingest once into `eval_{uuid}` session → each case runs the real compiled subgraph → DeepEval metrics → timestamped report + `latest.json`.

## Step/badge protocol (backend ↔ frontend)

A step emitted by a graph node is either a **plain string** (detail line) or a **badge dict**:

```python
{"badge": "llm",    "model": str, "in": int, "out": int, "cached": int, "cost": float, "ms": int}
{"badge": "qdrant", "mode": str, "candidates": int, "embed_ms": int, "qdrant_ms": int}
{"badge": "cohere", "model": str, "pairs": int, "top": int, "ms": int}
{"badge": "chunks", "n": int}   # internal only — consumed by PipelineMetrics, never shown
```

SSE detail events carry `step` (string) or `badge` (object); `ThinkingBlock.tsx` renders badges as chips directly from the object. **There is no string parsing of badges anywhere — keep it that way.**

SSE event types: `thinking_step` · `thinking_step_output` · `thinking_done` · `context_limit` · `token` · `pipeline_summary` · `done` · `error`; upload: `progress` · `complete` · `error`.

## Critical rules (learned the hard way)

1. **All tunables live in `backend/config.py`.** Never hardcode model names, thresholds, chunk sizes, or paths elsewhere.
2. **Run from project root.** All imports are absolute (`from backend.X import Y`).
3. **`gpt-5-nano` is a reasoning model.** It burns completion tokens on internal CoT before any visible output. `MAX_TOKENS` is 4000; never set it below ~1000 or answers come back empty (`completion_tokens` maxed, `content=""`).
4. **`_emit` and the audit registry are plain dicts keyed by `session_id`, not `threading.local()`.** LangGraph nodes run on thread-pool workers — thread-locals set on the event-loop thread are invisible there. `_emit(session_id, step)` also keeps concurrent sessions isolated; don't reintroduce a broadcast.
5. **`_doc_id()` must include chunk type** in the hash key (`session:type:source:page:idx`) — otherwise the summary doc (idx=0, page=0) collides with the first text chunk and silently overwrites it in Qdrant.
6. **Qdrant `metadata.type` needs a KEYWORD payload index** for filtered scrolls (`get_document_summary`). `_ensure_payload_index()` creates it at collection creation and lazily on read.
7. **`QDRANT_VECTOR_SIZE = 1536`** (text-embedding-3-small). Changing the embedder means deleting and re-ingesting every existing collection.
8. **BM25 sparse indexes clean `page_content` only**; the contextual `Title/H1-H3` prefix is applied at dense-embed time in `vector_store.add_documents`. Don't bake prefixes into stored chunk text.
9. **`document_summary` chunks are excluded from answer context** (`generate_answer_node`) and from eval `retrieval_context` — they semantically overlap everything and poison both answers and contextual metrics.
10. **DeepEval must run as a subprocess** (`routes/evaluation.py`): its `evaluate()` owns an event loop and installs a SIGINT handler; in-process it kills uvicorn. `subprocess.run` via `asyncio.to_thread` (Windows SelectorEventLoop doesn't support `create_subprocess_exec`).
11. **Perf metric names** (`"Latency (s)"`, `"Retrieval Time (s)"`, `"Generation Time (s)"`, `"Token Usage"`) must stay stable — `compare.py` and `reports/baseline.json` match on them.
12. **`turnMetaMapRef` lives in `App.tsx`**, not `ChatWindow` — `ChatWindow` has `key={active.id}` and unmounts on session switch, destroying any inner refs. History reloads re-hydrate messages from this map.
13. **Corporate proxy intercepts SSL** — Groq and Cohere calls use `httpx.Client(verify=False)`. Expect the same need for any new direct HTTP integration.
14. **SQLite session registry uses a fresh WAL connection per call** (`session_store._connect`). A shared connection across FastAPI's thread pool caused `sqlite3.InterfaceError`.
15. **ChatPromptTemplate-style escaping**: literal `{`/`}` in prompt templates must be `{{`/`}}`.
16. Session delete is a **6-step cascade** (registry → Qdrant collection → checkpoints → uploaded PDF → PNGs → upload status). Keep new per-session state in that list.

## Key config values (verify in `config.py` before relying on them)

| Constant | Value | Note |
|---|---|---|
| `CHAT_MODEL` / `VISION_MODEL` | `gpt-5-nano` | generation + VLM captioning |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 1536-dim, OpenAI API |
| `COHERE_RERANK_MODEL` | `rerank-v4.0-fast` | ~300 ms (local BGE was 23–27 s) |
| `HYBRID_PREFETCH_LIMIT` / `RERANK_TOP_N` | 10 / 5 | candidates → final chunks |
| `RELEVANCY_SCORE_THRESHOLD` | 0.45 | CRAG gate on top rerank score |
| `MAX_REWRITE_COUNT` | 1 | one query-rewrite retry |
| `CHILD_CHUNK_SIZE` / `PARENT_CHUNK_SIZE` | 256 / 512 | tiktoken cl100k_base tokens |
| `MAX_TOKENS` / `MAX_TOKENS_VISION` | 4000 / 1500 | reasoning-model headroom |
| `PIPELINE_MODE` | `hybrid` | `low_cost` \| `hybrid` \| `high_quality` (table routing) |
| `EVAL_MODEL` | `gpt-4.1-mini` | DeepEval judge |
| Context guard | warn 10k / stop 20k chars | in `routes/chat.py` |

## Environment (.env)

`OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `COHERE_API_KEY`, `GROQ_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING`, `PIPELINE_MODE`.

## Known gaps

- Chat history is checkpointed but **not passed to the answer LLM** — each answer is independent; pronoun follow-ups ("what about the second one?") don't resolve. Planned: last-N-turns between system and user messages (keeps the cacheable system prefix).
- No auth on the API; upload progress percentages are staged, not measured; the UI pipeline-mode selector isn't wired to the upload endpoint.
