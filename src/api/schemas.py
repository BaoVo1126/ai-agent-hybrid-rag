from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    strategy: str = "function_calling"
    session_id: str | None = None  # omit to run stateless (no history saved)


class ChatStep(BaseModel):
    step_type: str
    content: str


class ChatResponse(BaseModel):
    final_answer: str
    strategy: str
    session_id: str | None
    latency_seconds: float
    tool_calls_made: int
    llm_calls_made: int
    steps: list[ChatStep]


class CreateSessionRequest(BaseModel):
    title: str | None = None


class SessionInfo(BaseModel):
    id: str
    title: str | None
    created_at: str


class SessionMessage(BaseModel):
    role: str
    content: str
    strategy: str | None
    created_at: str
