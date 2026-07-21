"""
FastAPI backend.

Endpoints:
  GET  /                        -> serves the web UI (web/index.html)
  GET  /api/health               -> liveness + mode (mock/real, memory/postgres, memory/pgvector)
  GET  /api/strategies           -> which agent strategies are available
  POST /api/upload               -> save a file into data/ and rebuild the index
  POST /api/sessions              -> create a new chat session (returns session_id)
  GET  /api/sessions              -> list existing sessions
  GET  /api/sessions/{id}/messages -> full history for one session
  POST /api/chat                 -> run one query through the chosen agent strategy
  POST /api/chat/stream           -> same, but as a real Server-Sent Events (SSE)
                                      stream (text/event-stream)

Streaming note: self_correcting_rag_agent.py implements true incremental
streaming (run_stream() yields each step the instant it's produced -- see
that file). The other three strategies (react/function_calling/plan_execute)
still run to completion first and then replay their trace step-by-step over
the same SSE connection (AgentStrategy.run_stream()'s default fallback in
core/interfaces.py) -- correct output, just not live yet. Give any of them
the same generator refactor self_correcting_rag_agent.py got when that
matters for your use case; nothing else needs to change, callers here only
depend on the run_stream() interface.

Session history: when a request includes session_id, both the user's query
and the agent's final answer are persisted via
db/chat_history.build_chat_history_store() (memory by default, Postgres +
Redis-cache when CHAT_HISTORY_BACKEND=postgres). Omit session_id for a
stateless one-off call -- nothing is written.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.agents.factory import get_agent
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStep,
    CreateSessionRequest,
    SessionInfo,
    SessionMessage,
)
from src.config import SETTINGS
from src.core.llm_client import get_llm_client
from src.db.chat_history import build_chat_history_store
from src.ingestion.indexer import build_index
from src.tools.registry import ToolRegistry, build_default_registry

app = FastAPI(title="AI Agent Lab", description="A framework-agnostic AI agent you can point at any document.")

_state: dict = {"registry": None, "llm": None, "history": None}


def _get_registry() -> ToolRegistry:
    if _state["registry"] is None:
        _state["registry"] = build_default_registry()
    return _state["registry"]


def _get_llm():
    if _state["llm"] is None:
        _state["llm"] = get_llm_client()
    return _state["llm"]


def _get_history():
    if _state["history"] is None:
        _state["history"] = build_chat_history_store()
    return _state["history"]


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_mode": "real" if SETTINGS.is_real_mode else "mock",
        "vector_backend": SETTINGS.vector_backend,
        "chat_history_backend": SETTINGS.chat_history_backend,
    }


@app.get("/api/strategies")
def strategies() -> dict:
    return {
        "strategies": ["react", "function_calling", "plan_execute", "self_correcting_rag"],
        "default": "function_calling",
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use .pdf, .txt, or .md")

    os.makedirs(SETTINGS.data_dir, exist_ok=True)
    dest_path = os.path.join(SETTINGS.data_dir, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Rebuild explicitly (train-once, serve-many) rather than re-indexing on
    # every chat request.
    build_index()
    _state["registry"] = None  # force re-pull of the fresh index on next use
    return {"status": "indexed", "filename": file.filename}


# ------------------------------------------------------------- sessions ---
@app.post("/api/sessions", response_model=SessionInfo)
def create_session(req: CreateSessionRequest) -> SessionInfo:
    history = _get_history()
    session_id = history.create_session(title=req.title)
    sessions = history.list_sessions()
    match = next((s for s in sessions if s["id"] == session_id), {"id": session_id, "title": req.title, "created_at": ""})
    return SessionInfo(**match)


@app.get("/api/sessions", response_model=list[SessionInfo])
def list_sessions() -> list[SessionInfo]:
    return [SessionInfo(**s) for s in _get_history().list_sessions()]


@app.get("/api/sessions/{session_id}/messages", response_model=list[SessionMessage])
def get_session_messages(session_id: str) -> list[SessionMessage]:
    return [SessionMessage(**m) for m in _get_history().get_messages(session_id)]


# ------------------------------------------------------------------ chat ---
def _persist_turn(session_id: str | None, query: str, answer: str, strategy: str) -> None:
    if not session_id:
        return
    history = _get_history()
    history.add_message(session_id, role="user", content=query)
    history.add_message(session_id, role="assistant", content=answer, strategy=strategy)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    agent = get_agent(req.strategy, _get_llm(), _get_registry())
    result = agent.run(req.query)
    _persist_turn(req.session_id, req.query, result.final_answer, req.strategy)
    return ChatResponse(
        final_answer=result.final_answer,
        strategy=req.strategy,
        session_id=req.session_id,
        latency_seconds=result.latency_seconds,
        tool_calls_made=result.tool_calls_made,
        llm_calls_made=result.llm_calls_made,
        steps=[ChatStep(step_type=s.step_type, content=s.content) for s in result.steps],
    )


def _sse_frame(event: str, data: dict) -> str:
    # Standard SSE wire format: an optional "event:" line, one or more
    # "data:" lines, then a blank line to terminate the frame. Using a
    # single JSON-encoded "data:" line per frame keeps parsing on the
    # client trivial (see web/app.js).
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    agent = get_agent(req.strategy, _get_llm(), _get_registry())

    async def event_source():
        loop = asyncio.get_event_loop()
        step_iter = agent.run_stream(req.query)
        final_answer = "(no answer produced)"

        while True:
            # agent.run_stream() is a plain sync generator (the underlying
            # LLM/tool calls are blocking, urllib-based -- see
            # core/llm_client.py); running its `next()` in a thread pool
            # keeps this coroutine from blocking the FastAPI event loop
            # while a step is being computed.
            step = await loop.run_in_executor(None, _next_or_none, step_iter)
            if step is None:
                break
            yield _sse_frame(step.step_type, {"step_type": step.step_type, "content": step.content})
            if step.step_type == "final_answer":
                final_answer = step.content

        _persist_turn(req.session_id, req.query, final_answer, req.strategy)
        yield _sse_frame("done", {"step_type": "done", "content": final_answer})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # disable nginx response buffering for real streaming in prod
    )


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "web", "index.html"))


_web_dir = os.path.join(os.path.dirname(__file__), "..", "..", "web")
if os.path.isdir(_web_dir):
    app.mount("/web", StaticFiles(directory=_web_dir), name="web")
