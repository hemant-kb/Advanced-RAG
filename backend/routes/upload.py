"""PDF upload endpoint — streams ingestion progress as SSE events.

Events:
  {"type": "progress", "message": "...", "progress": 0.0-1.0}
  {"type": "complete", "message": "...", "chunks_created": N, "breakdown": {...}}
  {"type": "error",    "message": "..."}
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langsmith import trace as ls_trace

from backend import session_store
from backend.config import LANGSMITH_PROJECT, UPLOAD_DIR
from backend.models import UploadStatus
from backend.rag.document_pipeline import ingest_pdf
from backend.routes.common import SSE_HEADERS, sse

router = APIRouter()

_upload_status: dict[str, UploadStatus] = {}


def clear_upload_status(session_id: str) -> None:
    """Drop the in-memory status entry (used by the session delete cascade)."""
    _upload_status.pop(session_id, None)


@router.post("/sessions/{session_id}/upload")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    """Upload a PDF and stream granular ingestion progress as SSE events."""
    session_store.get(session_id)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    dest = Path(UPLOAD_DIR) / f"{session_id}_{file.filename}"
    contents = await file.read()
    with dest.open("wb") as f:
        f.write(contents)

    filename = file.filename

    async def _progress_stream() -> AsyncIterator[str]:
        yield sse({"type": "progress", "message": f"Saved {filename}. Starting ingestion…", "progress": 0.05})
        _upload_status[session_id] = UploadStatus(status="processing", progress=0.05, message="Starting…")

        try:
            # Step 1 — parse
            yield sse({"type": "progress", "message": "Parsing PDF structure (pages, tables, images)…", "progress": 0.15})
            _upload_status[session_id] = UploadStatus(status="processing", progress=0.15, message="Parsing PDF…")

            # Step 2 — extract + embed (the heavy work)
            yield sse({"type": "progress", "message": "Extracting text, tables and images in parallel…", "progress": 0.30})
            _upload_status[session_id] = UploadStatus(status="processing", progress=0.30, message="Extracting…")

            with ls_trace(
                name=f"ingest_pdf | session={session_id[:8]}",
                run_type="chain",
                project_name=LANGSMITH_PROJECT,
                metadata={"session_id": session_id, "filename": filename},
            ):
                result = await asyncio.to_thread(ingest_pdf, str(dest), session_id)

            # Step 3 — embedding (already done inside ingest_pdf, but surface it)
            yield sse({"type": "progress", "message": "Embedding chunks into Qdrant…", "progress": 0.85})
            _upload_status[session_id] = UploadStatus(status="processing", progress=0.85, message="Embedding…")

            session_store.set_document(session_id, filename)

            breakdown = {
                "text":  result.get("text_chunks", 0),
                "table": result.get("table_chunks", 0),
                "image": result.get("image_chunks", 0),
            }
            total = result["chunks_created"]
            parts = [f"{v} {k}" for k, v in breakdown.items() if v]
            msg = f"Ready — {total} chunks ingested ({', '.join(parts)})"

            _upload_status[session_id] = UploadStatus(
                status="complete", progress=1.0,
                message=msg, chunks_created=total,
            )
            yield sse({
                "type": "complete",
                "message": msg,
                "chunks_created": total,
                "breakdown": breakdown,
            })

        except Exception as e:
            _upload_status[session_id] = UploadStatus(status="error", progress=0.0, message=str(e))
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        _progress_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/sessions/{session_id}/upload/status", response_model=UploadStatus)
def upload_status(session_id: str):
    session_store.get(session_id)
    return _upload_status.get(
        session_id, UploadStatus(status="pending", progress=0.0, message="")
    )
