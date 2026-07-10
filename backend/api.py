"""
FastAPI app assembly — routers, CORS, lifespan.

Persistence:
  - Session registry: SQLite (backend/session_store.py)
  - Conversation state: LangGraph's SQLite checkpointer (thread_id = session id)
  - PDF chunks: Qdrant, one collection per session

Routes live in backend/routes/:
  sessions.py   — CRUD, auto-name, history, cascading delete
  upload.py     — PDF upload with SSE ingestion progress
  chat.py       — SSE-streamed RAG chat (event protocol documented there)
  evaluation.py — DeepEval subprocess trigger
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend import session_store
from backend.config import CHECKPOINT_DB, UPLOAD_DIR
from backend.graph import build_graph
from backend.routes import chat, evaluation, sessions, upload

Path(UPLOAD_DIR).mkdir(exist_ok=True)
session_store.init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        yield


app = FastAPI(title="RAG Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(evaluation.router)


@app.get("/health")
def health():
    return {"status": "ok"}
