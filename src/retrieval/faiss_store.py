"""
FAISSStore -- in-memory dense vector index using FAISS. The
VECTOR_BACKEND=faiss counterpart to PGVectorStore: real semantic
(embedding-based) search without needing a running Postgres instance,
closing the accuracy gap the plain "memory" backend has -- BM25 and TF-IDF
are both purely lexical, neither understands synonyms or paraphrasing (see
retrieval/tfidf.py's docstring).

Trades PGVectorStore's durability/multi-process sharing for zero external
infrastructure: everything lives in one process's RAM and persists to a
single pickle file on disk (see retrieval/hybrid_faiss.py +
ingestion/indexer.py's build_faiss_index/load_faiss_index) -- the same
"pickle file, zero setup" deal the original BM25+TFIDF `memory` backend
already has, just with a real ANN index instead of a plain numpy matrix.

Same lazy-import + graceful-fallback convention as retrieval/reranker.py
and retrieval/embeddings.py: importing this module never requires `faiss`
to be installed, only actually constructing/using a FAISSStore does.

Uses IndexFlatIP (exact inner-product search, no approximation) rather than
an approximate index like IVF/HNSW: at the "one document, a few thousand
chunks" scale this project targets, brute-force is both fast enough and
exactly correct, and it skips the extra "train the index" step approximate
indexes need before they can be queried. Embeddings come pre-normalized
(embeddings.py: normalize_embeddings=True) so inner product IS cosine
similarity, matching the "higher score = more relevant" convention every
other retriever in this project already follows.
"""

from __future__ import annotations

import numpy as np

from src.core.interfaces import Document


class FAISSStore:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._index = None
        self.documents: list[Document] = []

    def _empty_index(self):
        import faiss  # optional dependency, imported lazily

        return faiss.IndexFlatIP(self.dimension)

    def build(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError(f"documents ({len(documents)}) and embeddings ({len(embeddings)}) length mismatch")
        self._index = self._empty_index()
        self.documents = documents
        if documents:
            vectors = np.asarray(embeddings, dtype="float32")
            self._index.add(vectors)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Document, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        query = np.asarray([query_embedding], dtype="float32")
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))
        results: list[tuple[Document, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS pads short result rows with -1
                continue
            results.append((self.documents[idx], float(score)))
        return results

    def __getstate__(self) -> dict:
        """FAISS index objects aren't picklable by the stdlib `pickle`
        module directly -- serialize to bytes via faiss.serialize_index so
        the whole HybridFaissRetriever (BM25 + this store) can still ride
        the same single-pickle-file persistence the original memory backend
        uses (ingestion/indexer.py)."""
        import faiss

        state = self.__dict__.copy()
        state["_index"] = faiss.serialize_index(self._index) if self._index is not None else None
        return state

    def __setstate__(self, state: dict) -> None:
        import faiss

        serialized = state.pop("_index")
        self.__dict__.update(state)
        self._index = faiss.deserialize_index(serialized) if serialized is not None else None
