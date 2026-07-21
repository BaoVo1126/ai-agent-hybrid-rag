"""
Chat session history storage.

Same dual-mode pattern used throughout this codebase (LLM_BACKEND
mock/ollama, VECTOR_BACKEND memory/postgres): a small interface with two
implementations, selected by SETTINGS.chat_history_backend, so api/main.py
never needs to know or care which one is active.

- MemoryChatHistoryStore: in-process dict. Zero setup, resets on restart.
  This is the default -- a fresh clone still "just works" with no database.
- PostgresChatHistoryStore: durable, shared across worker processes/replicas.
  Reads go through Redis first when available (cache/redis_client.py);
  writes go to Postgres first (source of truth) and then update the cache,
  so a cache miss or a Redis outage never loses data, only read speed.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TypedDict


class MessageDict(TypedDict):
    role: str
    content: str
    strategy: str | None
    created_at: str


class ChatHistoryStore(ABC):
    @abstractmethod
    def create_session(self, title: str | None = None) -> str: ...

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str, strategy: str | None = None) -> None: ...

    @abstractmethod
    def get_messages(self, session_id: str, limit: int | None = None) -> list[MessageDict]: ...

    @abstractmethod
    def list_sessions(self) -> list[dict]: ...


class MemoryChatHistoryStore(ChatHistoryStore):
    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._messages: dict[str, list[MessageDict]] = {}

    def create_session(self, title: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {"id": session_id, "title": title, "created_at": _now_iso()}
        self._messages[session_id] = []
        return session_id

    def add_message(self, session_id: str, role: str, content: str, strategy: str | None = None) -> None:
        if session_id not in self._messages:
            self._sessions[session_id] = {"id": session_id, "title": None, "created_at": _now_iso()}
            self._messages[session_id] = []
        self._messages[session_id].append(
            {"role": role, "content": content, "strategy": strategy, "created_at": _now_iso()}
        )

    def get_messages(self, session_id: str, limit: int | None = None) -> list[MessageDict]:
        messages = self._messages.get(session_id, [])
        return messages[-limit:] if limit else list(messages)

    def list_sessions(self) -> list[dict]:
        return sorted(self._sessions.values(), key=lambda s: s["created_at"], reverse=True)


class PostgresChatHistoryStore(ChatHistoryStore):
    def __init__(self, cache=None) -> None:
        from src.db.session import init_db

        init_db()  # idempotent: CREATE TABLE IF NOT EXISTS, safe to call on every process start
        self.cache = cache

    def create_session(self, title: str | None = None) -> str:
        from src.db.models import ChatSession
        from src.db.session import new_session

        with new_session() as db:
            session = ChatSession(title=title)
            db.add(session)
            db.commit()
            return session.id

    def add_message(self, session_id: str, role: str, content: str, strategy: str | None = None) -> None:
        from src.db.models import ChatMessage, ChatSession
        from src.db.session import new_session

        with new_session() as db:
            if db.get(ChatSession, session_id) is None:
                db.add(ChatSession(id=session_id))
            db.add(ChatMessage(session_id=session_id, role=role, content=content, strategy=strategy))
            db.commit()

        if self.cache is not None:
            self.cache.invalidate(_cache_key(session_id))

    def get_messages(self, session_id: str, limit: int | None = None) -> list[MessageDict]:
        if self.cache is not None:
            cached = self.cache.get_json(_cache_key(session_id))
            if cached is not None:
                return cached[-limit:] if limit else cached

        from src.db.models import ChatMessage
        from src.db.session import new_session

        with new_session() as db:
            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            messages: list[MessageDict] = [
                {
                    "role": row.role,
                    "content": row.content,
                    "strategy": row.strategy,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

        if self.cache is not None:
            self.cache.set_json(_cache_key(session_id), messages)

        return messages[-limit:] if limit else messages

    def list_sessions(self) -> list[dict]:
        from src.db.models import ChatSession
        from src.db.session import new_session

        with new_session() as db:
            rows = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
            return [{"id": row.id, "title": row.title, "created_at": row.created_at.isoformat()} for row in rows]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_key(session_id: str) -> str:
    return f"chat_history:{session_id}"


def build_chat_history_store() -> ChatHistoryStore:
    from src.config import SETTINGS

    if SETTINGS.chat_history_backend == "postgres":
        cache = None
        if SETTINGS.redis_url:
            from src.cache.redis_client import RedisCache

            cache = RedisCache(SETTINGS.redis_url, ttl_seconds=SETTINGS.session_cache_ttl_seconds)
        return PostgresChatHistoryStore(cache=cache)
    return MemoryChatHistoryStore()
