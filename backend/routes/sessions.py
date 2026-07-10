"""Session endpoints — CRUD, auto-naming, history, cascading delete."""
from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Request

from backend import session_store
from backend.config import CHECKPOINT_DB, IMAGE_STORE_DIR, OPENAI_API_KEY, UPLOAD_DIR
from backend.models import (
    AutoNameRequest,
    CreateSessionRequest,
    RenameSessionRequest,
    SessionInfo,
)
from backend.rag.vector_store import delete_collection
from backend.routes.upload import clear_upload_status

logger = logging.getLogger("rag.requests")

router = APIRouter()


@router.post("/sessions", response_model=SessionInfo)
def create_session(req: CreateSessionRequest):
    return session_store.create(req.name)


@router.get("/sessions", response_model=list[SessionInfo])
def list_sessions():
    return session_store.list_all()


@router.get("/sessions/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    return session_store.get(session_id)


@router.patch("/sessions/{session_id}", response_model=SessionInfo)
def rename_session(session_id: str, req: RenameSessionRequest):
    session_store.get(session_id)
    session_store.rename(session_id, req.name)
    return session_store.get(session_id)


# ── Auto-naming ─────────────────────────────────────────────────

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


@router.post("/sessions/{session_id}/auto-name")
async def auto_name_session(session_id: str, req: AutoNameRequest):
    """Generate a short session name from the user's first message using an LLM."""
    session_store.get(session_id)
    try:
        resp = await _get_openai_client().chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (3-5 words max) for a chat based on the user's first message. "
                        "Return ONLY the title, no punctuation, no quotes, no explanation."
                    ),
                },
                {"role": "user", "content": req.message[:500]},
            ],
            max_tokens=20,
            temperature=0.3,
        )
        name = resp.choices[0].message.content.strip().strip('"').strip("'")
        if name:
            session_store.rename(session_id, name)
        return {"name": name}
    except Exception as e:
        logger.warning(f"Auto-name failed: {e}")
        return {"name": ""}


# ── Delete (cascading) ──────────────────────────────────────────

def _delete_checkpoints(session_id: str) -> None:
    """Delete all LangGraph state rows for this session from checkpoints.db."""
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    try:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
        conn.execute("DELETE FROM writes      WHERE thread_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


def _delete_upload_file(session_id: str) -> None:
    """Delete the PDF file uploaded for this session, if any."""
    for f in Path(UPLOAD_DIR).glob(f"{session_id}_*"):
        f.unlink(missing_ok=True)


def _delete_image_store(session_id: str) -> None:
    """Delete all saved PNGs for this session."""
    img_dir = Path(IMAGE_STORE_DIR) / session_id
    if img_dir.exists():
        shutil.rmtree(img_dir, ignore_errors=True)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    session_store.get(session_id)
    # 1. Session registry
    session_store.delete(session_id)
    # 2. Qdrant vector collection
    delete_collection(session_id)
    # 3. LangGraph conversation state (checkpoints + writes)
    _delete_checkpoints(session_id)
    # 4. Uploaded PDF file
    _delete_upload_file(session_id)
    # 5. Saved image PNGs
    _delete_image_store(session_id)
    # 6. In-memory upload status
    clear_upload_status(session_id)
    return {"deleted": session_id}


# ── History ─────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/history")
def get_history(session_id: str, request: Request):
    session_store.get(session_id)
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = request.app.state.graph.get_state(config=config)
        messages = (state.values or {}).get("messages", []) if state else []
    except Exception:
        messages = []

    # Keep only human messages and the last assistant message per turn.
    # LangGraph can accumulate multiple AIMessages per turn (agent retries,
    # query rewrites, etc.) — we show only the final answer.
    result = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.type == "human":
            content = m.content if isinstance(m.content, str) else str(m.content)
            result.append({"role": "user", "content": content})
            i += 1
        else:
            # Consume consecutive non-human messages, keep the last non-empty one.
            j = i
            while j < len(messages) and messages[j].type != "human":
                j += 1
            final_answer = ""
            for am in messages[i:j]:
                content = am.content if isinstance(am.content, str) else str(am.content)
                if content.strip():
                    final_answer = content
            if final_answer:
                result.append({"role": "assistant", "content": final_answer})
            i = j

    return {"messages": result}
