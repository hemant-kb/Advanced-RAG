"""Shared helpers for SSE-streaming routes."""
from __future__ import annotations

import json

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse(data: dict) -> str:
    """Encode one Server-Sent Events data line."""
    return f"data: {json.dumps(data)}\n\n"
