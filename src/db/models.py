"""
SQLAlchemy models for chat session history.

Two tables:
  chat_sessions -- one row per conversation ("session_id" from the API)
  chat_messages -- one row per turn (user query or agent answer), FK'd to a
                   session, ordered by created_at

Kept deliberately small: no user accounts, no auth here -- session_id is a
client-generated UUID (see api/main.py POST /api/sessions). Add a users
table and a foreign key on ChatSession the day this needs real multi-user
auth; nothing here blocks that.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_str() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid_str)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    strategy: Mapped[str | None] = mapped_column(String, nullable=True)  # which agent strategy answered, for assistant messages
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    session: Mapped[ChatSession] = relationship(back_populates="messages")
