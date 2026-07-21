from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import main as api_main
from src.core.llm_client import MockLLMClient


def _client_with_test_registry(tool_registry):
    # Inject the in-memory test registry/LLM instead of letting the API build
    # its own from data/ -- keeps these tests independent of any real file.
    # Also reset "history" each time: api_main._state is a module-level dict
    # shared across every test in this file/process, so without resetting it
    # a session created by one test would still be visible to the next.
    api_main._state["registry"] = tool_registry
    api_main._state["llm"] = MockLLMClient()
    api_main._state["history"] = None
    return TestClient(api_main.app)


def test_health_endpoint(tool_registry):
    client = _client_with_test_registry(tool_registry)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["llm_mode"] == "mock"
    # Added when chat history + the pgvector backend were introduced --
    # both default to "memory" so a plain `pytest` run here (no
    # POSTGRES_DSN/REDIS_URL set) never touches a real database.
    assert body["vector_backend"] == "memory"
    assert body["chat_history_backend"] == "memory"


def test_strategies_endpoint(tool_registry):
    client = _client_with_test_registry(tool_registry)
    response = client.get("/api/strategies")
    assert response.status_code == 200
    strategies = response.json()["strategies"]
    assert "react" in strategies
    assert "self_correcting_rag" in strategies  # added alongside the reranker/self-correction upgrade


def test_chat_endpoint_returns_final_answer(tool_registry):
    client = _client_with_test_registry(tool_registry)
    response = client.post("/api/chat", json={"query": "What is 3 + 4?", "strategy": "function_calling"})
    assert response.status_code == 200
    data = response.json()
    assert "7" in data["final_answer"]
    assert data["strategy"] == "function_calling"


def test_chat_endpoint_persists_history_when_session_id_given(tool_registry):
    client = _client_with_test_registry(tool_registry)
    session = client.post("/api/sessions", json={}).json()

    client.post("/api/chat", json={"query": "What is 3 + 4?", "strategy": "function_calling", "session_id": session["id"]})

    messages = client.get(f"/api/sessions/{session['id']}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "What is 3 + 4?"


def test_chat_endpoint_without_session_id_persists_nothing(tool_registry):
    client = _client_with_test_registry(tool_registry)
    client.post("/api/chat", json={"query": "What is 3 + 4?", "strategy": "function_calling"})
    # No session_id -> stateless call, no session should have been created for it
    assert client.get("/api/sessions").json() == []


def _parse_sse(raw_text: str) -> list[dict]:
    import json

    frames = []
    for chunk in raw_text.strip().split("\n\n"):
        if not chunk.strip():
            continue
        data_line = next((line for line in chunk.split("\n") if line.startswith("data:")), None)
        if data_line:
            frames.append(json.loads(data_line[len("data:") :].strip()))
    return frames


def test_chat_stream_endpoint_yields_real_sse_frames(tool_registry):
    client = _client_with_test_registry(tool_registry)
    response = client.post("/api/chat/stream", json={"query": "What is 3 + 4?", "strategy": "react"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = _parse_sse(response.text)
    assert len(frames) >= 1
    assert frames[-1]["step_type"] == "done"
