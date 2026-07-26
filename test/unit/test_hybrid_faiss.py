from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("faiss")

from src.core.interfaces import Document  # noqa: E402
from src.retrieval.bm25 import tokenize  # noqa: E402
from src.retrieval.embeddings import EmbeddingModel  # noqa: E402
from src.retrieval.hybrid_faiss import HybridFaissRetriever  # noqa: E402

_DIM = 32


class DeterministicHashEmbedder(EmbeddingModel):
    """A tiny, fully offline stand-in for a real sentence-transformers
    model: a hashing-trick bag-of-words vector. Deterministic and needs no
    network/model download, but still gives documents that share vocabulary
    a higher cosine similarity than unrelated ones -- enough to test fusion
    behavior meaningfully without depending on a real embedding model in
    unit tests."""

    def __init__(self) -> None:
        super().__init__("deterministic-hash-embedder")

    @property
    def dimension(self) -> int:
        return _DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = np.zeros(_DIM, dtype="float32")
            for token in tokenize(text):
                vec[hash(token) % _DIM] += 1.0
            norm = np.linalg.norm(vec)
            vectors.append((vec / norm if norm > 0 else vec).tolist())
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@pytest.fixture
def faiss_retriever(sample_documents) -> HybridFaissRetriever:
    retriever = HybridFaissRetriever(embedder=DeterministicHashEmbedder())
    retriever.fit(sample_documents)
    return retriever


def test_fit_indexes_all_documents(faiss_retriever, sample_documents):
    assert len(faiss_retriever.documents) == len(sample_documents)
    assert faiss_retriever.store.documents == sample_documents


def test_search_returns_semantically_relevant_result_first(faiss_retriever):
    results = faiss_retriever.search("agent tools reasoning loop", top_k=2)
    assert results
    top_doc, _score = results[0]
    assert "agent" in top_doc.text.lower()


def test_search_on_empty_corpus_does_not_crash():
    retriever = HybridFaissRetriever(embedder=DeterministicHashEmbedder())
    retriever.fit([])
    assert retriever.search("anything", top_k=3) == []


def test_pickle_round_trip_preserves_search_behavior(faiss_retriever):
    """HybridFaissRetriever is persisted via pickle the same way the
    original memory backend is (ingestion/indexer.py) -- this locks in that
    the FAISS index, BM25 index, and embedder all survive that round trip."""
    restored: HybridFaissRetriever = pickle.loads(pickle.dumps(faiss_retriever))

    original = faiss_retriever.search("evaluation metrics precision recall", top_k=2)
    restored_results = restored.search("evaluation metrics precision recall", top_k=2)
    assert [d.id for d, _ in original] == [d.id for d, _ in restored_results]
