"""
Chat endpoint — streams RAG graph execution as Server-Sent Events.

Event kinds:
  {"type": "thinking_step", "step": "...", "node": "node_name", "output": null}
    — a top-level pipeline step (node === node name from NODE_LABELS)
  {"type": "thinking_step", "step": "...", "node": "detail", "parent_node": "node_name"}
    — a plain-text detail line emitted by a rag_graph node via _emit()
  {"type": "thinking_step", "badge": {...}, "node": "detail", "parent_node": "node_name"}
    — a structured badge (llm / qdrant / cohere) rendered as a metrics chip
  {"type": "thinking_step_output", "node": "node_name", "output": "done"}
    — marks a node step as completed (emitted on on_chain_end)
  {"type": "thinking_done", "duration_ms": N}
  {"type": "context_limit", "message": "...", "warn_only": bool}
  {"type": "token",            "content": "..."}
  {"type": "pipeline_summary", ...aggregated metrics...}
  {"type": "done",             "answer": "..."}
  {"type": "error",            "message": "..."}
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from backend import session_store
from backend.config import LANGSMITH_PROJECT
from backend.models import ChatRequest
from backend.rag.audit import QueryAudit
from backend.rag.rag_graph import StepPayload, set_query_audit, set_step_callback
from backend.routes.common import SSE_HEADERS, sse

logger = logging.getLogger("rag.requests")

router = APIRouter()

# Context limit guard: warn at 10k chars of history, hard-stop at 20k.
CONTEXT_WARN_CHARS  = 10_000
CONTEXT_LIMIT_CHARS = 20_000

NODE_LABELS = {
    "rag":             "Starting RAG pipeline…",
    "entry":           "Routing query…",
    "retrieval":       "Retrieving from vector store…",
    "relevancy_check": "Checking relevance…",
    "query_rewrite":   "Rewriting query…",
    "generate_answer": "Generating answer…",
    "summary_node":    "Retrieving document summary…",
}


class PipelineMetrics:
    """Accumulates per-request metrics from the badge dicts emitted by rag_graph."""

    def __init__(self):
        self.llm_calls         = 0
        self.vector_searches   = 0
        self.chunks_retrieved  = 0
        self.chunks_reranked   = 0
        self.chunks_used       = 0
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.cached_tokens     = 0
        self.cost_usd          = 0.0
        self.qdrant_ms         = 0
        self.rerank_ms         = 0
        self.embed_ms          = 0

    def record(self, step: StepPayload) -> StepPayload | None:
        """
        Fold one step into the counters.
        Returns the payload to surface as a detail line in the thinking box,
        or None for internal-only badges.
        """
        if isinstance(step, str):
            return step
        kind = step.get("badge")
        if kind == "llm":
            self.llm_calls         += 1
            self.prompt_tokens     += int(step.get("in", 0) or 0)
            self.completion_tokens += int(step.get("out", 0) or 0)
            self.cached_tokens     += int(step.get("cached", 0) or 0)
            self.cost_usd          += float(step.get("cost", 0) or 0)
            return step
        if kind == "qdrant":
            self.vector_searches  += 1
            self.chunks_retrieved += int(step.get("candidates", 0) or 0)
            self.qdrant_ms        += int(step.get("qdrant_ms", 0) or 0)
            self.embed_ms         += int(step.get("embed_ms", 0) or 0)
            return step
        if kind == "cohere":
            self.chunks_reranked += int(step.get("top", 0) or 0)
            self.rerank_ms       += int(step.get("ms", 0) or 0)
            return step
        if kind == "chunks":
            self.chunks_used = int(step.get("n", 0) or 0)
            return None  # internal badge — not shown in thinking box
        return None  # unknown badge type — hide

    def summary_event(self, total_ms: int) -> dict:
        return {
            "type": "pipeline_summary",
            "total_ms":          total_ms,
            "llm_calls":         self.llm_calls,
            "vector_searches":   self.vector_searches,
            "chunks_retrieved":  self.chunks_retrieved,
            "chunks_reranked":   self.chunks_reranked,
            "chunks_used":       self.chunks_used,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens":     self.cached_tokens,
            "cost_usd":          round(self.cost_usd, 6),
            "qdrant_ms":         self.qdrant_ms,
            "rerank_ms":         self.rerank_ms,
            "embed_ms":          self.embed_ms,
        }


async def _stream_chat(graph, session_id: str, message: str) -> AsyncIterator[str]:
    from langchain_core.tracers.langchain import LangChainTracer
    request_id = str(uuid.uuid4())

    tracer = LangChainTracer(project_name=LANGSMITH_PROJECT)
    config = {
        "configurable": {"thread_id": session_id},
        "run_name": f"chat | session={session_id[:8]}",
        "tags": ["chat_request"],
        "metadata": {"session_id": session_id, "request_id": request_id},
        "callbacks": [tracer],
    }

    input_state = {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "query": message,
        # Reset per-turn transient state so a new question always starts fresh.
        "retrieval_attempts": 0,
        "rewrite_count": 0,
        "retrieved_docs": [],
        "is_relevant": None,
        "answer": None,
    }

    try:
        config_check = {"configurable": {"thread_id": session_id}}
        state_snapshot = graph.get_state(config=config_check)
        prior_messages = (state_snapshot.values or {}).get("messages", []) if state_snapshot else []
        total_chars = sum(len(str(m.content)) for m in prior_messages)
        if total_chars >= CONTEXT_LIMIT_CHARS:
            yield sse({"type": "context_limit", "message": "This conversation has reached its context limit. Please start a new chat to continue."})
            return
        if total_chars >= CONTEXT_WARN_CHARS:
            yield sse({"type": "context_limit", "message": "This conversation is getting long. Consider starting a new chat soon.", "warn_only": True})
    except Exception:
        pass

    t_start       = time.perf_counter()
    t_think_start = time.perf_counter()
    status        = "success"
    pm            = PipelineMetrics()

    # Queue for steps emitted synchronously from rag_graph nodes via the callback.
    step_queue: list[StepPayload] = []

    def _on_step(step: StepPayload) -> None:
        step_queue.append(step)

    # Install per-request callback and query audit.
    set_step_callback(_on_step, session_id)
    _q_audit = QueryAudit(session_id=session_id, query=message)
    _q_audit.save_query()
    set_query_audit(session_id, _q_audit)

    # Tracks the current node so detail steps are parented correctly.
    # KEY FIX: we update last_node_step ONLY after draining the step_queue from
    # the previous node (on_chain_end), not at on_chain_start of the next node.
    last_node_step: str | None = None

    async def _drain_queue():
        """Yield all pending detail steps attached to last_node_step."""
        while step_queue:
            payload = pm.record(step_queue.pop(0))
            if payload is None:
                continue
            event = {
                "type": "thinking_step",
                "node": "detail",
                "output": None,
                "parent_node": last_node_step,
            }
            if isinstance(payload, str):
                event["step"] = payload
            else:
                event["badge"] = payload
            yield sse(event)

    try:
        last_node_step = "start"

        acc = ""
        final_answer = ""

        async for event in graph.astream_events(input_state, config=config, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")

            if kind in ("on_chain_start", "on_chain_end", "on_retriever_start", "on_retriever_end"):
                logger.info("[stream_events] %s name=%r queue_len=%d", kind, name, len(step_queue))

            # on_retriever_start/end fire for @traceable(run_type="retriever") nodes
            # (retrieval_node); treat them the same as on_chain_start/end.
            is_node_start = kind in ("on_chain_start", "on_retriever_start")
            is_node_end   = kind in ("on_chain_end",   "on_retriever_end")

            if is_node_start and name in NODE_LABELS:
                # Flush any steps the previous node pushed BEFORE announcing the new node.
                async for sse_line in _drain_queue():
                    yield sse_line
                yield sse({"type": "thinking_step", "step": NODE_LABELS[name], "node": name, "output": None})
                last_node_step = name

            elif is_node_end:
                # Drain steps emitted by this node before marking it done.
                async for sse_line in _drain_queue():
                    yield sse_line
                if name in ("LangGraph", "rag"):
                    output = event.get("data", {}).get("output", {}) or {}
                    final_answer = output.get("answer", "") or acc
                if name in NODE_LABELS:
                    yield sse({"type": "thinking_step_output", "node": name, "output": "done"})

            elif kind == "on_chat_model_stream":
                # Drain any pending steps before streaming tokens.
                async for sse_line in _drain_queue():
                    yield sse_line
                chunk = event["data"].get("chunk")
                token = getattr(chunk, "content", "") or ""
                if token:
                    acc += token
                    yield sse({"type": "token", "content": token})

        # Final drain after graph completes.
        async for sse_line in _drain_queue():
            yield sse_line

        think_ms = int((time.perf_counter() - t_think_start) * 1000)

        # Emit pipeline summary before thinking_done so frontend has it before collapse.
        yield sse(pm.summary_event(think_ms))
        yield sse({"type": "thinking_done", "duration_ms": think_ms})
        yield sse({"type": "done", "answer": final_answer or acc})

    except Exception as e:
        status = "error"
        yield sse({"type": "error", "message": str(e)})

    finally:
        set_step_callback(None, session_id)
        set_query_audit(session_id, None)
        total_ms = int((time.perf_counter() - t_start) * 1000)

        log_record = {
            "request_id":        request_id,
            "session_id":        session_id,
            "query":             message[:200],
            "retrieved_chunks":  pm.chunks_retrieved,
            "vector_searches":   pm.vector_searches,
            "llm_calls":         pm.llm_calls,
            "prompt_tokens":     pm.prompt_tokens,
            "completion_tokens": pm.completion_tokens,
            "cost_usd":          round(pm.cost_usd, 6),
            "total_time_ms":     total_ms,
            "status":            status,
        }
        logger.info(json.dumps(log_record))


@router.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest, request: Request):
    session = session_store.get(session_id)
    if not session.has_document:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded. Please upload a PDF before asking questions.",
        )
    return StreamingResponse(
        _stream_chat(request.app.state.graph, session_id, req.message),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
