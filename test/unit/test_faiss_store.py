from __future__ import annotations

import pickle

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from src.core.interfaces import Document  # noqa: E402
from src.retrieval.faiss_store import FAISSStore  # noqa: E402


def _unit_vector(seed: int, dim: int = 8) -> list[float]:
    rng = np.random.RandomState(seed)
    v = rng.rand(dim).astype("float32")
    return (v / np.linalg.norm(v)).tolist()


def test_build_and_search_returns_closest_vector_first():
    docs = [Document(id="a", text="doc a", metadata={}), Document(id="b", text="doc b", metadata={})]
    embeddings = [_unit_vector(1), _unit_vector(2)]
    store = FAISSStore(dimension=8)
    store.build(docs, embeddings)

    results = store.search(embeddings[0], top_k=2)
    assert results[0][0].id == "a"
    assert results[0][1] >= results[1][1]


def test_search_on_empty_store_returns_empty_list():
    store = FAISSStore(dimension=8)
    store.build([], [])
    assert store.search(_unit_vector(1), top_k=3) == []


def test_build_rejects_mismatched_lengths():
    docs = [Document(id="a", text="doc a", metadata={})]
    with pytest.raises(ValueError):
        FAISSStore(dimension=8).build(docs, [])


def test_pickle_round_trip_preserves_search_results():
    """FAISS index objects aren't natively picklable -- FAISSStore defines
    __getstate__/__setstate__ (serialize_index/deserialize_index) so it can
    ride inside a pickled HybridFaissRetriever, same persistence as the
    original memory backend. This locks that in."""
    docs = [Document(id="a", text="doc a", metadata={}), Document(id="b", text="doc b", metadata={})]
    embeddings = [_unit_vector(1), _unit_vector(2)]
    store = FAISSStore(dimension=8)
    store.build(docs, embeddings)

    restored: FAISSStore = pickle.loads(pickle.dumps(store))
    assert [d.id for d in restored.documents] == ["a", "b"]

    original_results = store.search(embeddings[0], top_k=2)
    restored_results = restored.search(embeddings[0], top_k=2)
    assert [d.id for d, _ in original_results] == [d.id for d, _ in restored_results]
