"""
Engine + session factory for the chat-history database.

The DSN is shared with the pgvector store (SETTINGS.postgres_dsn) --
in the simplest deployment this is the same Postgres instance holding both
the chat_sessions/chat_messages tables and the document_chunks table.
Nothing forces that: point them at two different databases if you'd rather
keep vector storage and chat history physically separate.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import SETTINGS
from src.db.models import Base

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(SETTINGS.postgres_dsn)
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def new_session() -> Session:
    return get_session_factory()()
