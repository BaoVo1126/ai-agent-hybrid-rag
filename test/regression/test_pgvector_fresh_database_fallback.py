"""
Regression test for bug #4 in docs/bugs-found.md.

HybridPGVectorRetriever.load_from_store() used to call store.fetch_all()
without first ensuring the table exists. On a brand-new database (first
deploy, nothing ingested yet) that raised a raw
sqlalchemy.exc.ProgrammingError ("relation does not exist") instead of the
RuntimeError that indexer.py's load_or_build_index() is written to catch
and fall back to a full build from -- so the "just works on first run"
promise silently broke for the postgres backend.

Requires a real reachable Postgres with the pgvector extension (set
POSTGRES_DSN, e.g. via the docker-compose in this repo) -- skipped
automatically otherwise, same convention as other infra-dependent tests.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("pgvector")

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql+psycopg2://agentlab:agentlab@localhost:5432/agentlab")


def _postgres_reachable() -> bool:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(POSTGRES_DSN)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _postgres_reachable(), reason="no reachable Postgres for this test")
def test_load_from_store_on_fresh_database_raises_runtime_error_not_programming_error():
    from sqlalchemy import create_engine, text

    from src.retrieval.hybrid_pgvector import HybridPGVectorRetriever
    from src.retrieval.pgvector_store import PGVectorStore

    engine = create_engine(POSTGRES_DSN)
    table_name = "test_regression_fresh_table"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))

    store = PGVectorStore(engine, dimension=8, table_name=table_name)

    class _FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

        def embed_one(self, text):
            return [0.0] * 8

    # Must raise RuntimeError (the type load_or_build_index() catches),
    # never a raw database exception -- that's the whole point of the fix.
    with pytest.raises(RuntimeError):
        HybridPGVectorRetriever.load_from_store(store, embedder=_FakeEmbedder())

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
