"""
HybridFaissRetriever -- BM25 (sparse) fused with FAISSStore (dense,
in-process) via the same reciprocal_rank_fusion() used by HybridRetriever
(BM25+TFIDF) and HybridPGVectorRetriever (BM25+pgvector). This is the "real
semantic search, zero external infrastructure" backend selected by
VECTOR_BACKEND=faiss (src/config.py, the default): no Postgres to run, but
-- unlike TF-IDF -- an actual embedding model doing the semantic side,
closing the "neither BM25 nor TF-IDF understands synonyms/paraphrasing" gap
called out in retrieval/tfidf.py's docstring.

TF-IDF itself is dropped here rather than kept as a third fusion input, for
the same reason hybrid_pgvector.py drops it: TF-IDF and dense embeddings
both exist to catch semantic similarity BM25's keyword-overlap scoring
misses, and the embedding model does that job strictly better -- so this is
a straight upgrade over HybridRetriever (BM25+TFIDF), not a 3-way fusion.

Persisted the same way as the original memory backend: one pickle file
(ingestion/indexer.py::build_faiss_index / load_faiss_index), just holding
a HybridFaissRetriever instead of a HybridRetriever. FAISSStore and
EmbeddingModel both define __getstate__/__setstate__ so that pickle file
never embeds the loaded FAISS C++ object or model weights directly -- see
those modules' docstrings.
"""

from __future__ import annotations

from src.core.interfaces import Document
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.fusion import reciprocal_rank_fusion


class HybridFaissRetriever:
    def __init__(self, embedder: EmbeddingModel | None = None, rrf_k: int = 60) -> None:
        self.embedder = embedder or EmbeddingModel()
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever()
        self.store: FAISSStore | None = None
        self.documents: list[Document] = []

    def fit(self, documents: list[Document]) -> None:
        self.documents = documents
        self.bm25.fit(documents)

        embeddings = self.embedder.embed([doc.text for doc in documents]) if documents else []
        self.store = FAISSStore(dimension=self.embedder.dimension)
        self.store.build(documents, embeddings)

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> list[tuple[Document, float]]:
        bm25_results = self.bm25.search(query, top_k=candidate_pool)

        vector_results: list[tuple[Document, float]] = []
        if self.store is not None:
            query_embedding = self.embedder.embed_one(query)
            vector_results = self.store.search(query_embedding, top_k=candidate_pool)

        return reciprocal_rank_fusion([bm25_results, vector_results], top_k=top_k, rrf_k=self.rrf_k)
