"""
RAG subgraph — purely retrieval-augmented, no direct-answer bypass:
  - All queries go through retrieval (no router)
  - Retrieval node searches directly — no agent LLM call for tool selection
  - CRAG: score-based relevancy check + one query rewrite on miss
  - Two-phase vision: text answer by default, vision LLM if image chunks retrieved

Graph flow:
  entry → retrieval → relevancy_check
                        ↓ relevant      → generate_answer
                        ↓ not relevant  → query_rewrite → retrieval (retry)
                        ↓ retry exhausted → generate_answer (fallback)

Step tracking: every node emits detailed thinking steps via a per-session
callback registered by api.py before each graph run. A step is either a plain
string (shown as a detail line) or a badge dict (rendered as a metrics chip):

  {"badge": "llm",    "model": str, "in": int, "out": int, "cached": int, "cost": float, "ms": int}
  {"badge": "qdrant", "mode": str, "candidates": int, "embed_ms": int, "qdrant_ms": int}
  {"badge": "cohere", "model": str, "pairs": int, "top": int, "ms": int}
  {"badge": "chunks", "n": int}   — internal only, consumed for pipeline metrics
"""
from __future__ import annotations

import logging
import time as _time
import traceback
from typing import Callable, Union

logger = logging.getLogger("rag.graph")

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langsmith import traceable

from backend.config import (
    CAPTION_ANSWER_PROMPT,
    CAPTION_ANSWER_SYSTEM,
    CHAT_MODEL,
    CHAT_MODEL_PRICE_CACHED,
    CHAT_MODEL_PRICE_INPUT,
    CHAT_MODEL_PRICE_OUTPUT,
    COHERE_RERANK_MODEL,
    MAX_REWRITE_COUNT,
    MAX_TOKENS,
    MAX_TOKENS_VISION,
    NEEDS_VISUAL_TOKEN,
    RELEVANCY_SCORE_THRESHOLD,
    VISION_MODEL,
)
from backend.rag.guardrails import check_prompt_injection
from backend.rag.vector_store import search as vs_search, get_document_summary

NO_CONTEXT_ANSWER = (
    "I couldn't find relevant information in the uploaded document "
    "to answer your question. You may want to rephrase or upload a different document."
)

# ── Summary intent detection ─────────────────────────────────────

_SUMMARY_KEYWORDS = {
    "summarise", "summarize", "summary", "summarization",
    "overview", "outline", "abstract",
    "what is this document", "what is this paper", "what is this pdf",
    "what does this document", "what does this paper", "what does this pdf",
    "tell me about this document", "tell me about this paper",
    "what is it about", "what's it about",
    "give me a summary", "give me an overview",
}


def _is_summary_query(query: str) -> bool:
    q = query.lower().strip()
    return any(kw in q for kw in _SUMMARY_KEYWORDS)


# ── Step callback (set by api.py per-request) ───────────────────
# Keyed by session_id so it works across threads (LangGraph nodes run on
# a thread pool, not the async event loop thread that calls set_step_callback)
# and so concurrent sessions never see each other's steps.

StepPayload = Union[str, dict]

_callback_registry: dict[str, Callable[[StepPayload], None]] = {}


def set_step_callback(cb: Callable[[StepPayload], None] | None, session_id: str = "") -> None:
    if cb is None:
        _callback_registry.pop(session_id, None)
    else:
        _callback_registry[session_id] = cb


def _emit(session_id: str, step: StepPayload) -> None:
    cb = _callback_registry.get(session_id)
    if cb is None:
        return
    try:
        cb(step)
    except Exception:
        pass


def _emit_llm_usage(session_id: str, response, model: str, elapsed_ms: int) -> None:
    """Emit an llm badge dict consumed by api.py for cost/token tracking."""
    usage = getattr(response, "usage_metadata", None) or {}
    in_tok  = usage.get("input_tokens", 0) or 0
    out_tok = usage.get("output_tokens", 0) or 0
    # OpenAI cached tokens live in response_metadata.token_usage.prompt_token_details
    resp_meta = getattr(response, "response_metadata", {}) or {}
    token_details = resp_meta.get("token_usage", {}) or {}
    prompt_details = token_details.get("prompt_token_details", {}) or {}
    cached_tok = prompt_details.get("cached_tokens", 0) or 0

    # Cost: cached tokens billed at lower rate
    billable_in = max(0, in_tok - cached_tok)
    cost = (
        billable_in  * CHAT_MODEL_PRICE_INPUT  / 1_000_000
        + cached_tok * CHAT_MODEL_PRICE_CACHED / 1_000_000
        + out_tok    * CHAT_MODEL_PRICE_OUTPUT / 1_000_000
    )
    _emit(session_id, {
        "badge": "llm", "model": model,
        "in": in_tok, "out": out_tok, "cached": cached_tok,
        "cost": round(cost, 6), "ms": elapsed_ms,
    })


# ── Query audit (set by api.py per-request) ──────────────────────
# Keyed by session_id — thread-safe for concurrent requests; LangGraph nodes
# may run on different threads than the caller, so thread-local doesn't work.

_audit_registry: dict[str, object] = {}


def set_query_audit(session_id: str, audit) -> None:
    if audit is None:
        _audit_registry.pop(session_id, None)
    else:
        _audit_registry[session_id] = audit


def _get_audit(session_id: str):
    return _audit_registry.get(session_id)


# ── LLM singletons ──────────────────────────────────────────────

chat_llm   = ChatOpenAI(model=CHAT_MODEL,   max_tokens=MAX_TOKENS)
vision_llm = ChatOpenAI(model=VISION_MODEL, max_tokens=MAX_TOKENS_VISION)


# ── State ───────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str
    retrieved_docs: list[Document]
    retrieval_attempts: int
    rewrite_count: int
    is_relevant: bool | None
    answer: str | None


# ── Retrieval node ─────────────────────────────────────────────

@traceable(name="rag_retrieval_node", run_type="retriever",
           metadata={"graph": "rag", "node": "retrieval"})
def retrieval_node(state: RAGState) -> dict:
    """Search directly using state["query"] — no agent LLM call needed."""
    query      = state["query"]
    session_id = state["session_id"]
    current_docs = list(state.get("retrieved_docs") or [])
    _audit = _get_audit(session_id)
    _is_rewrite = state.get("rewrite_count", 0) > 0

    _emit(session_id, f"Hybrid search (dense + BM25 + rerank): \"{query[:80]}\"")

    _prefetch_holder: list = []
    _reranked_holder: list = []
    _timing_holder: dict = {}

    def _on_timing(t: dict) -> None:
        _timing_holder.update(t)
        if _audit and "dense_vector" in t:
            try:
                _audit.save_dense_vector(t["dense_vector"], t.get("embed_ms", 0))
                if t.get("sparse_vec"):
                    _audit.save_sparse_vector(t["sparse_vec"])
            except Exception:
                logger.error("[retrieval_node] audit vector save failed\n%s",
                             traceback.format_exc())

    try:
        docs = vs_search(
            query=query,
            session_id=session_id,
            _prefetch_cb=lambda d: _prefetch_holder.extend(d),
            _reranked_cb=lambda d: _reranked_holder.extend(d),
            _timing_cb=_on_timing,
        )
    except Exception as e:
        logger.error("[retrieval_node] vs_search raised for query=%r session=%s\n%s",
                     query[:80], session_id, traceback.format_exc())
        _emit(session_id, f"Retrieval error: {e}")
        return {"retrieved_docs": current_docs}

    if _audit:
        try:
            if _prefetch_holder:
                _audit.save_prefetch_results(_prefetch_holder)
            if _reranked_holder:
                if _is_rewrite:
                    _audit.save_rerank_after_rewrite(_reranked_holder)
                else:
                    _audit.save_reranked_results(_reranked_holder)
        except Exception:
            pass

    new_docs = list(current_docs)
    if docs:
        new_docs.extend(docs)
        type_counts: dict[str, int] = {}
        for d in docs:
            t = d.metadata.get("type", "text")
            type_counts[t] = type_counts.get(t, 0) + 1
        breakdown = ", ".join(f"{v} {k}" for k, v in sorted(type_counts.items()))
        _emit(session_id, f"Retrieved {len(docs)} chunks ({breakdown})")

        qdrant_ms   = _timing_holder.get("qdrant_ms", 0)
        embed_ms    = _timing_holder.get("embed_ms", 0)
        rerank_ms   = _timing_holder.get("rerank_ms", 0)
        search_mode = _timing_holder.get("mode", "hybrid_rrf")
        n_prefetch  = len(_prefetch_holder)
        _emit(session_id, {
            "badge": "qdrant", "mode": search_mode, "candidates": n_prefetch,
            "embed_ms": embed_ms, "qdrant_ms": qdrant_ms,
        })

        if _prefetch_holder:
            _emit(session_id, f"── Prefetch pool — {n_prefetch} candidates (before reranking) ──")
            for rank, d in enumerate(_prefetch_holder, 1):
                src   = d.metadata.get("source", "?")
                page  = d.metadata.get("page", "?")
                dtype = d.metadata.get("type", "text")
                _emit(session_id, f"  {rank:2d}. [{dtype}] p.{page} · {src} — \"{d.page_content[:80].strip()}…\"")

        _emit(session_id, {
            "badge": "cohere", "model": COHERE_RERANK_MODEL,
            "pairs": n_prefetch, "top": len(docs), "ms": rerank_ms,
        })

        _emit(session_id, f"── After reranking — top {len(docs)} chunks ──")
        for rank, d in enumerate(docs, 1):
            src   = d.metadata.get("source", "?")
            page  = d.metadata.get("page", "?")
            dtype = d.metadata.get("type", "text")
            score = d.metadata.get("rerank_score")
            score_str = f"score={score:.3f}" if score is not None else "score=n/a"
            _emit(session_id, f"  {rank}. [{dtype}] p.{page} · {score_str} · {src} — \"{d.page_content[:80].strip()}…\"")
    else:
        _emit(session_id, "No relevant chunks found")

    return {"retrieved_docs": new_docs}


# ── Relevancy + rewrite nodes ───────────────────────────────────

@traceable(name="rag_relevancy_check_node", run_type="chain",
           metadata={"graph": "rag", "node": "relevancy_check"})
def relevancy_check_node(state: RAGState) -> dict:
    """Score-based relevancy — no LLM call. Uses top Cohere rerank score as signal."""
    session_id = state["session_id"]
    docs = state.get("retrieved_docs") or []
    if not docs:
        _emit(session_id, "Relevancy check: no chunks retrieved → not relevant")
        return {"is_relevant": False}

    top_score = docs[0].metadata.get("rerank_score") or 0.0
    is_relevant = top_score >= RELEVANCY_SCORE_THRESHOLD
    verdict = "✅ relevant" if is_relevant else "❌ not relevant"
    _emit(session_id, f"Relevancy check: top score={top_score:.3f} (threshold={RELEVANCY_SCORE_THRESHOLD}) → {verdict}")

    _audit = _get_audit(session_id)
    if _audit:
        try:
            reason = f"top rerank score {top_score:.3f} {'≥' if is_relevant else '<'} threshold {RELEVANCY_SCORE_THRESHOLD}"
            _audit.save_relevancy_check(is_relevant, reason)
        except Exception:
            pass
    return {"is_relevant": is_relevant}


QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant. The previous query failed to retrieve "
    "relevant chunks. Rewrite using more specific terminology, domain keywords, "
    "or a narrower sub-question. Return ONLY the rewritten query."
)


@traceable(name="rag_query_rewrite_node", run_type="chain",
           metadata={"graph": "rag", "node": "query_rewrite", "model": CHAT_MODEL})
def query_rewrite_node(state: RAGState) -> dict:
    session_id = state["session_id"]
    original = state["query"]
    _emit(session_id, f"Original query: \"{original[:80]}\"")
    _t0 = _time.perf_counter()
    response = chat_llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original}\n\nRewrite it."},
    ])
    _ms = int((_time.perf_counter() - _t0) * 1000)
    _emit_llm_usage(session_id, response, CHAT_MODEL, _ms)
    rewritten = response.content.strip()
    _emit(session_id, f"Rewritten query: \"{rewritten[:120]}\"")
    _audit = _get_audit(session_id)
    if _audit:
        try:
            _audit.save_rewritten_query(rewritten)
        except Exception:
            pass
    return {
        "query": rewritten,
        "retrieved_docs": [],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "is_relevant": None,
    }


# ── Answer generation (caption-first, vision fallback) ──────────
#
# Phase 1: chat_llm answers using caption/text context.
#          Image chunks contribute their VLM-generated caption (page_content).
#          If the caption is enough, we're done — no vision LLM needed.
#
# Phase 2: if chat_llm signals [NEEDS_VISUAL], load the original PNG from
#          image_path on disk and send to vision_llm with the original question.
#          This avoids redundant VLM calls when the caption already answers.

def _doc_context(doc: Document) -> str:
    """Return parent content for text chunks, raw content for tables/images."""
    if doc.metadata.get("type") == "text":
        return doc.metadata.get("parent_content") or doc.page_content
    return doc.page_content


def _load_image_b64(image_path: str) -> str | None:
    """Load PNG from disk and return as base64 string."""
    import base64
    from pathlib import Path
    try:
        return base64.b64encode(Path(image_path).read_bytes()).decode()
    except Exception:
        return None


def _build_vision_message(query: str, docs: list[Document]) -> dict:
    """Build a multimodal message loading images from disk paths."""
    content: list[dict] = []
    text_parts: list[str] = []
    image_parts: list[dict] = []

    for d in docs:
        if d.metadata.get("type") in ("image", "chart"):
            text_parts.append(f"[Image on page {d.metadata.get('page')}: {d.page_content}]")
            img_path = d.metadata.get("image_path")
            if img_path:
                b64 = _load_image_b64(img_path)
                if b64:
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
        else:
            text_parts.append(_doc_context(d))

    context_block = "\n\n---\n\n".join(text_parts)
    content.append({
        "type": "text",
        "text": (
            f"Use the context below (including images) to answer the question precisely.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}"
        ),
    })
    content.extend(image_parts)
    return {"role": "user", "content": content}


@traceable(name="rag_generate_answer_node", run_type="llm",
           metadata={"graph": "rag", "node": "generate_answer", "model": CHAT_MODEL})
def generate_answer_node(state: RAGState) -> dict:
    session_id = state["session_id"]
    query = state["query"]
    docs  = state.get("retrieved_docs") or []

    if not docs or (state.get("is_relevant") is False and state.get("rewrite_count", 0) >= MAX_REWRITE_COUNT):
        _emit(session_id, "No relevant chunks found — returning fallback answer")
        return {"answer": NO_CONTEXT_ANSWER, "messages": [AIMessage(content=NO_CONTEXT_ANSWER)]}

    # Exclude document_summary chunks — those are for the summary fast-path only,
    # not for per-question answer generation (they pollute context with unrelated facts).
    answer_docs = [d for d in docs if d.metadata.get("type") != "document_summary"]
    if not answer_docs:
        answer_docs = docs  # fallback: if somehow all chunks are summaries, use them

    # Dedup: two child chunks from the same parent expand to identical parent text.
    # Keep only the first occurrence per unique context string.
    _seen_ctx: set[str] = set()
    deduped_docs: list = []
    for d in answer_docs:
        ctx = _doc_context(d)
        if ctx not in _seen_ctx:
            _seen_ctx.add(ctx)
            deduped_docs.append(d)
    answer_docs = deduped_docs

    # Phase 1: answer from captions + text (cheap, no vision LLM)
    context = "\n\n---\n\n".join(_doc_context(d) for d in answer_docs)
    _audit  = _get_audit(session_id)
    if _audit:
        try:
            _audit.save_context_sent(context)
        except Exception:
            pass

    # ── Context detail (what the LLM will see) ─────────────────────
    _emit(session_id, f"── Context sent to LLM ({len(answer_docs)} chunks) ──")
    for rank, d in enumerate(answer_docs, 1):
        src    = d.metadata.get("source", "?")
        page   = d.metadata.get("page", "?")
        dtype  = d.metadata.get("type", "text")
        score  = d.metadata.get("rerank_score")
        score_str = f"score={score:.3f}" if score is not None else ""
        ctx_text = (_doc_context(d))[:120].strip()
        _emit(session_id, f"  {rank}. [{dtype}] p.{page} · {src}{' · ' + score_str if score_str else ''} — \"{ctx_text}…\"")

    _emit(session_id, {"badge": "chunks", "n": len(answer_docs)})
    _emit(session_id, f"Phase 1: answering from captions/text ({len(answer_docs)} chunks)…")
    user_msg = CAPTION_ANSWER_PROMPT.format(context=context, query=query)
    _t0 = _time.perf_counter()
    phase1_response = chat_llm.invoke([
        {"role": "system", "content": CAPTION_ANSWER_SYSTEM},
        {"role": "user",   "content": user_msg},
    ])
    _ms = int((_time.perf_counter() - _t0) * 1000)
    _emit_llm_usage(session_id, phase1_response, CHAT_MODEL, _ms)
    answer = phase1_response.content.strip() or NO_CONTEXT_ANSWER

    if _audit:
        try:
            _audit.save_phase1_answer(answer)
        except Exception:
            pass

    # Phase 2: vision fallback — only if chat_llm signals it needs the actual image
    has_visuals = any(d.metadata.get("type") in ("image", "chart") for d in answer_docs)
    if has_visuals and NEEDS_VISUAL_TOKEN in answer:
        _emit(session_id, "Phase 2: caption insufficient — loading original image(s) for vision LLM…")
        if _audit:
            try:
                _img_paths = [
                    d.metadata["image_path"] for d in answer_docs
                    if d.metadata.get("type") in ("image", "chart") and d.metadata.get("image_path")
                ]
                _audit.save_vision_triggered(_img_paths)
            except Exception:
                pass
        msg = _build_vision_message(query, answer_docs)
        _t0 = _time.perf_counter()
        phase2_response = vision_llm.invoke([msg])
        _ms = int((_time.perf_counter() - _t0) * 1000)
        _emit_llm_usage(session_id, phase2_response, VISION_MODEL, _ms)
        answer = phase2_response.content.strip()
    elif has_visuals:
        _emit(session_id, "Phase 2: not triggered — Phase 1 caption answer was sufficient")

    if _audit:
        try:
            _audit.save_final_answer(answer)
            _audit.finish()
        except Exception:
            pass

    return {"answer": answer, "messages": [AIMessage(content=answer)]}


# ── Summary node ────────────────────────────────────────────────

@traceable(name="rag_summary_node", run_type="chain",
           metadata={"graph": "rag", "node": "summary_node"})
def summary_node(state: RAGState) -> dict:
    session_id = state["session_id"]
    _emit(session_id, "Summary intent detected — retrieving stored document summary…")
    summary = get_document_summary(session_id)
    if summary:
        _emit(session_id, f"Stored summary found ({len(summary)} chars)")
        answer = summary
    else:
        _emit(session_id, "No stored summary found — falling back to retrieval")
        answer = (
            "A pre-generated summary is not available for this document yet. "
            "Please re-upload the document to generate one."
        )
    _audit = _get_audit(session_id)
    if _audit:
        try:
            _audit.save_final_answer(answer)
            _audit.finish()
        except Exception:
            pass
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


# ── Routing helpers ─────────────────────────────────────────────

def entry_routing(state: RAGState) -> str:
    """Short-circuits injection-blocked and summary queries before retrieval."""
    session_id = state.get("session_id", "")
    if state.get("answer"):  # injection guardrail already set the answer
        return END
    if _is_summary_query(state.get("query", "")):
        _emit(session_id, "Route: summary fast-path (no vector search needed)")
        return "summary_node"
    _emit(session_id, "Route: retrieval pipeline")
    return "retrieval"


def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < MAX_REWRITE_COUNT:
        return "query_rewrite"
    return "generate_answer"


# ── Builder ─────────────────────────────────────────────────────

def _entry_node(state: RAGState) -> dict:
    """Entry point: run guardrails before routing to retrieval or summary."""
    query = state.get("query", "")
    rejection = check_prompt_injection(query)
    if rejection:
        _emit(state.get("session_id", ""), "Guardrail: prompt injection detected — blocking query")
        return {"answer": rejection, "is_relevant": False, "rewrite_count": 99}
    return {}


def build_rag_subgraph():
    graph = StateGraph(RAGState)
    graph.add_node("entry",           _entry_node)
    graph.add_node("retrieval",       retrieval_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite",   query_rewrite_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("summary_node",    summary_node)

    graph.set_entry_point("entry")

    graph.add_conditional_edges(
        "entry", entry_routing,
        {
            END:            END,
            "retrieval":    "retrieval",
            "summary_node": "summary_node",
        },
    )

    graph.add_edge("retrieval", "relevancy_check")
    graph.add_conditional_edges(
        "relevancy_check", after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite",   "retrieval")
    graph.add_edge("generate_answer", END)
    graph.add_edge("summary_node",    END)

    return graph
