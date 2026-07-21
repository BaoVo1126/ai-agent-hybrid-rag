"""
PGVectorStore -- dense vector storage/search backed by a real Postgres
database (pgvector extension), replacing the pickle file
(ingestion/indexer.py's INDEX_FILE) as the place chunk embeddings live.

Why this over the pickle file: a pickle is a single in-memory blob loaded
fresh into every process -- fine for a single-process demo, but it means
(a) every API worker/replica needs its own full copy in RAM, (b) there's no
way to add/update documents without rebuilding and redistributing the whole
file, and (c) nothing else (a second service, an admin script) can query the
index without importing this exact Python retriever class. A real database
fixes all three: multiple processes share one source of truth, updates are
incremental (`upsert_chunks`), and it's queryable with plain SQL if needed.

Schema: one row per chunk, `embedding` is a pgvector column. Cosine distance
(`<=>`) is used for search since embeddings are L2-normalized at encode time
(retrieval/embeddings.py: `normalize_embeddings=True`), which makes cosine
distance and it's much cheaper `1 - dot product` interchangeable in ranking
order.
"""

from __future__ import annotations

import json

from sqlalchemy import Column, MetaData, String, Table, Text, delete, select, text
from sqlalchemy.engine import Engine

from src.core.interfaces import Document

_metadata = MetaData()


def _table(table_name: str, dimension: int) -> Table:
    from pgvector.sqlalchemy import Vector  # optional dependency, imported lazily

    return Table(
        table_name,
        _metadata,
        Column("id", String, primary_key=True),
        Column("text", Text, nullable=False),
        Column("doc_metadata", Text, nullable=False),  # JSON-encoded; see note below
        Column("embedding", Vector(dimension), nullable=False),
        extend_existing=True,
    )


class PGVectorStore:
    """
    doc_metadata is stored as a JSON *string* column rather than JSONB on
    purpose: it keeps this module's only hard dependency the `pgvector`
    package (for the Vector column type) instead of also coupling to
    Postgres-specific JSONB operators, so the same table shape would port
    to another pgvector-compatible engine with minimal changes.
    """

    def __init__(self, engine: Engine, dimension: int, table_name: str = "document_chunks") -> None:
        self.engine = engine
        self.dimension = dimension
        self.table = _table(table_name, dimension)

    def ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            _metadata.create_all(conn, tables=[self.table])

    def clear(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(self.table))

    def upsert_chunks(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError(f"documents ({len(documents)}) and embeddings ({len(embeddings)}) length mismatch")
        if not documents:
            return

        rows = [
            {
                "id": doc.id,
                "text": doc.text,
                "doc_metadata": json.dumps(doc.metadata),
                "embedding": emb,
            }
            for doc, emb in zip(documents, embeddings)
        ]
        with self.engine.begin() as conn:
            # Simple delete-then-insert upsert: fine at chunk-corpus scale
            # (thousands, not millions, of rows) and keeps this method
            # dialect-agnostic instead of relying on Postgres' ON CONFLICT.
            ids = [r["id"] for r in rows]
            conn.execute(delete(self.table).where(self.table.c.id.in_(ids)))
            conn.execute(self.table.insert(), rows)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Document, float]]:
        distance = self.table.c.embedding.cosine_distance(query_embedding)
        stmt = select(self.table.c.id, self.table.c.text, self.table.c.doc_metadata, distance.label("distance")).order_by(distance).limit(top_k)

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()

        results = []
        for row in rows:
            doc = Document(id=row.id, text=row.text, metadata=json.loads(row.doc_metadata))
            similarity = 1.0 - float(row.distance)  # cosine distance -> similarity, for a score that reads like the other retrievers'
            results.append((doc, similarity))
        return results

    def fetch_all(self) -> list[Document]:
        """Reads every stored chunk back out, with no vector search involved
        -- used at process startup to rebuild the in-memory BM25 index from
        whatever is already durably stored in Postgres, instead of
        re-reading/re-chunking/re-embedding the original files every time a
        worker process starts (see hybrid_pgvector.py `load_from_store`)."""
        stmt = select(self.table.c.id, self.table.c.text, self.table.c.doc_metadata)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [Document(id=row.id, text=row.text, metadata=json.loads(row.doc_metadata)) for row in rows]

    def count(self) -> int:
        from sqlalchemy import func

        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(self.table)).scalar_one()
