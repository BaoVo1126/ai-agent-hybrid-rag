"""
Thin Redis wrapper used as a read-through cache in front of Postgres for
chat history (db/chat_history.py PostgresChatHistoryStore).

Same "never hard-fail the app over an optional dependency" philosophy as
retrieval/reranker.py's missing-sentence-transformers fallback: if Redis is
unreachable (not installed, wrong URL, container not up yet), every method
here just no-ops / returns None instead of raising, and a warning is
printed once. The chat still works -- it just always reads from Postgres
instead of getting the cache speed-up. A cache is allowed to fail open;
the source of truth (Postgres) is not.
"""

from __future__ import annotations

import json


class RedisCache:
    def __init__(self, url: str, ttl_seconds: int = 3600) -> None:
        self.url = url
        self.ttl_seconds = ttl_seconds
        self._client = None
        self._warned = False
        self._available = True

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._available:
            return None
        try:
            import redis  # optional dependency, imported lazily

            client = redis.from_url(self.url, socket_connect_timeout=1, socket_timeout=1)
            client.ping()
            self._client = client
            return client
        except Exception as exc:  # broad on purpose: ImportError, ConnectionError, TimeoutError all mean "no cache"
            self._available = False
            if not self._warned:
                print(f"[redis_cache] unavailable ({exc.__class__.__name__}: {exc}) -- reads/writes will hit Postgres directly.")
                self._warned = True
            return None

    def get_json(self, key: str):
        client = self._get_client()
        if client is None:
            return None
        try:
            raw = client.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception:
            return None

    def set_json(self, key: str, value) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            client.set(key, json.dumps(value), ex=self.ttl_seconds)
        except Exception:
            pass

    def invalidate(self, key: str) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            client.delete(key)
        except Exception:
            pass
