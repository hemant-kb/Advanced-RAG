"""All Pydantic schemas used across the application."""
from typing import Literal, Optional

from pydantic import BaseModel


# ── CRAG relevancy ───────────────────────────────────────────────

class RelevancyDecision(BaseModel):
    is_relevant: bool
    reason: str


# ── API request/response schemas ────────────────────────────────

class CreateSessionRequest(BaseModel):
    name: Optional[str] = None


class SessionInfo(BaseModel):
    id: str
    name: str
    created_at: str
    has_document: bool
    document_name: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class RenameSessionRequest(BaseModel):
    name: str


class AutoNameRequest(BaseModel):
    message: str


class EvaluateRequest(BaseModel):
    pdf_path: str
    goldens_file: Optional[str] = None  # defaults to GOLDENS_FILE from config
    max_cases: Optional[int] = None     # limit test cases, e.g. 1 for smoke-test


class UploadStatus(BaseModel):
    status: Literal["pending", "processing", "complete", "error"]
    progress: float = 0.0
    message: str = ""
    chunks_created: int = 0
