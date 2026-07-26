"""
Regression-style test for the same "missing optional dependency must
degrade gracefully, not crash" class of bug as
tests/regression/test_reranker_missing_dependency_fallback.py (bug #3 in
docs/bugs-found.md): VECTOR_BACKEND=faiss needs faiss-cpu and
sentence-transformers installed, and since it's now the default backend
(src/config.py), a fresh clone without those packages installed must still
"just work" via a fallback to the BM25+TFIDF-only memory backend, exactly
like build_default_registry() already does for the reranker.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

from src.core.interfaces import Document
from src.ingestion import indexer
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.hybrid import HybridRetriever


class _StubSentenceTransformer:
    """Stands in for a real (network-downloaded) SentenceTransformer model
    so the "faiss-cpu missing" test below exercises the actual
    ImportError-from-inside-FAISSStore fallback path without needing
    network access to Hugging Face Hub for the embedding side, which isn't
    what this test is about."""

    def get_sentence_embedding_dimension(self) -> int:
        return 8

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        return np.ones((len(texts), 8), dtype="float32")


@pytest.fixture
def faiss_settings(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample.txt").write_text(
        "Ravens are highly intelligent birds that use tools to solve problems."
    )
    index_dir = tmp_path / ".index"

    fake_settings = dataclasses.replace(
        indexer.SETTINGS,
        vector_backend="faiss",
        data_dir=str(data_dir),
        index_dir=str(index_dir),
    )
    monkeypatch.setattr(indexer, "SETTINGS", fake_settings)
    return fake_settings


def test_load_or_build_index_falls_back_to_memory_when_faiss_missing(faiss_settings, monkeypatch):
    # Stub out the embedding model load so this test's network dependency
    # is exactly what it's testing (faiss) and nothing else -- the fallback
    # in load_or_build_index() only fires once EmbeddingModel.embed()/​
    # dimension has already succeeded and FAISSStore itself tries `import
    # faiss`, same order of operations as the real code path.
    monkeypatch.setattr(EmbeddingModel, "_load", lambda self: _StubSentenceTransformer())
    # Forces `import faiss` (used inside FAISSStore) to raise ImportError
    # regardless of whether it's actually installed in the environment
    # running this test -- a None entry in sys.modules is the standard way
    # to simulate "this import fails" without uninstalling anything.
    monkeypatch.setitem(sys.modules, "faiss", None)

    retriever = indexer.load_or_build_index()

    # Must NOT raise ModuleNotFoundError -- should gracefully fall back to
    # building the BM25+TFIDF memory index instead of crashing on startup.
    assert isinstance(retriever, HybridRetriever)
    results = retriever.search("intelligent birds using tools", top_k=1)
    assert results and "Ravens" in results[0][0].text


def test_load_or_build_index_falls_back_when_sentence_transformers_missing(faiss_settings, monkeypatch):
    # Here the ImportError should come from EmbeddingModel._load() itself
    # (`from sentence_transformers import SentenceTransformer`), before any
    # network call -- no stub needed.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    retriever = indexer.load_or_build_index()

    assert isinstance(retriever, HybridRetriever)
    results = retriever.search("intelligent birds", top_k=1)
    assert results
