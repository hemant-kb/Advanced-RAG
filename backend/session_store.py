"""
SQLite-backed session registry.

One row per chat session: (id, name, created_at, document_name).
Conversation state itself lives in LangGraph's checkpointer (checkpoints.db);
PDF chunks live in Qdrant. This table is only the session directory.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException

from backend.config import SESSIONS_DB
from backend.models import SessionInfo


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SESSIONS_DB, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                document_name TEXT
            )
        """)


def _row_to_session(row) -> SessionInfo:
    return SessionInfo(
        id=row[0],
        name=row[1],
        created_at=row[2],
        has_document=bool(row[3]),
        document_name=row[3],
    )


def create(name: str | None) -> SessionInfo:
    session_id = str(uuid.uuid4())
    name = name or f"New Chat {datetime.now().strftime('%H:%M')}"
    created = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, name, created_at, document_name) VALUES (?, ?, ?, NULL)",
            (session_id, name, created),
        )
    return SessionInfo(id=session_id, name=name, created_at=created, has_document=False)


def list_all() -> list[SessionInfo]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, document_name FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_session(r) for r in rows]


def get(session_id: str) -> SessionInfo:
    """Return the session or raise 404."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, created_at, document_name FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Session not found")
    return _row_to_session(row)


def rename(session_id: str, name: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sessions SET name=? WHERE id=?", (name, session_id))


def set_document(session_id: str, doc_name: str | None) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET document_name=? WHERE id=?",
            (doc_name, session_id),
        )


def delete(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
