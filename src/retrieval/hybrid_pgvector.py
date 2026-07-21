"""
HybridPGVectorRetriever -- the production retrieval backend: BM25 (sparse,
still in-memory -- see note below) fused with PGVectorStore (dense,
Postgres-backed) via the same reciprocal_rank_fusion() used by the original
HybridRetriever (BM25 + TF-IDF).

Why BM25 stays in-memory while the vector side moved to Postgres: BM25 is
lexical/keyword matching, which needs the *entire* term-document matrix to
rank properly -- there's no meaningful way to "page" it from a database
without re-implementing something like Postgres full-text search (a real
option later, but out of scope for this pass). It's also cheap: rebuilding
it from the chunk texts already stored in PGVectorStore takes milliseconds
even for a few thousand chunks, which is what `HybridPGVectorRetriever.fit()`
does at startup instead of persisting it separately. TF-IDF (the second
sparse signal in the original HybridRetriever) is dropped here rather than
kept as a third fusion input: it and the new dense embeddings both exist to
catch *semantic* similarity BM25's keyword-overlap scoring misses, and the
embedding model does that job strictly better (it understands synonyms and
paraphrasing; TF-IDF only understands shared words) -- so it's a straight
upgrade, not a 3-way fusion, matching the BM25+dense-vector pattern used in
most real hybrid-search systems.
"""

from __future__ import annotations

from src.core.interfaces import Document
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.pgvector_store import PGVectorStore


class HybridPGVectorRetriever:
    def __init__(self, store: PGVectorStore, embedder: EmbeddingModel | None = None, rrf_k: int = 60) -> None:
        self.store = store
        self.embedder = embedder or EmbeddingModel()
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever()
        self.documents: list[Document] = []

    def fit(self, documents: list[Document]) -> None:
        self.documents = documents
        self.bm25.fit(documents)

        self.store.ensure_schema()
        self.store.clear()
        embeddings = self.embedder.embed([doc.text for doc in documents])
        self.store.upsert_chunks(documents, embeddings)

    @classmethod
    def load_from_store(cls, store: PGVectorStore, embedder: EmbeddingModel | None = None) -> "HybridPGVectorRetriever":
        """Rebuilds the in-memory BM25 index from chunks already durably
        stored in Postgres -- no re-embedding, no re-reading the original
        files. This is what a freshly-started API worker calls; `fit()`
        above is only for the explicit ingest step (scripts/build_index.py).

        Calls ensure_schema() first: on a brand-new database the table
        doesn't exist yet, and without this the query below would raise a
        raw psycopg2/SQLAlchemy `ProgrammingError` ("relation does not
        exist") instead of the RuntimeError that indexer.py's
        load_or_build_index() is written to catch and fall back to a full
        build from -- caught by actually running this against a fresh
        database (see docs/bugs-found.md #4)."""
        store.ensure_schema()
        instance = cls(store=store, embedder=embedder)
        documents = store.fetch_all()
        if not documents:
            raise RuntimeError(
                "No chunks found in the pgvector table. Run the ingest step first "
                "(e.g. `python scripts/build_index.py`) with VECTOR_BACKEND=postgres set."
            )
        instance.documents = documents
        instance.bm25.fit(documents)
        return instance

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> list[tuple[Document, float]]:
        bm25_results = self.bm25.search(query, top_k=candidate_pool)

        query_embedding = self.embedder.embed_one(query)
        vector_results = self.store.search(query_embedding, top_k=candidate_pool)

        return reciprocal_rank_fusion([bm25_results, vector_results], top_k=top_k, rrf_k=self.rrf_k)
