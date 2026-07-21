"""
Indexer: separates index-building from serving, same "train-once, serve-many"
principle introduced in chatbot-rag's build_index.py. Run
`python scripts/build_index.py` after dropping a file into data/, then the
API/CLI/benchmark all load the already-built index instead of rebuilding it
on every startup.

Two backends now, chosen by SETTINGS.vector_backend (same dual-mode pattern
as LLM_BACKEND mock/ollama):

- "memory" (default): the original path. Persistence is a pickle file since
  the retrievers are plain Python/numpy objects -- zero setup, works on
  first clone, but doesn't share state across processes.
- "postgres": chunks + embeddings are stored durably in Postgres via
  pgvector (retrieval/pgvector_store.py, retrieval/hybrid_pgvector.py).
  Multiple API workers/replicas share the same index; add-a-document is an
  incremental upsert instead of a full rebuild-and-redistribute.
"""

from __future__ import annotations

import os
import pickle

from src.config import SETTINGS
from src.ingestion.chunking import chunk_documents
from src.ingestion.loaders import load_directory
from src.retrieval.hybrid import HybridRetriever

INDEX_FILE = "hybrid_index.pkl"


# ---------------------------------------------------------------- memory ---
def build_index(data_dir: str | None = None, index_dir: str | None = None) -> HybridRetriever:
    data_dir = data_dir or SETTINGS.data_dir
    index_dir = index_dir or SETTINGS.index_dir

    documents = load_directory(data_dir)
    if not documents:
        raise RuntimeError(
            f"No supported files (.pdf/.txt/.md) found in '{data_dir}'. "
            "Drop your file there first, e.g. data/AI Engineering.pdf"
        )

    chunks = chunk_documents(documents, chunk_size=SETTINGS.chunk_size, overlap=SETTINGS.chunk_overlap)
    retriever = HybridRetriever()
    retriever.fit(chunks)

    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, INDEX_FILE), "wb") as f:
        pickle.dump(retriever, f)

    return retriever


def load_index(index_dir: str | None = None) -> HybridRetriever:
    index_dir = index_dir or SETTINGS.index_dir
    path = os.path.join(index_dir, INDEX_FILE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"No index found at '{path}'. Run `python scripts/build_index.py` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------------------- postgres ---
def _pgvector_store():
    from sqlalchemy import create_engine

    from src.retrieval.pgvector_store import PGVectorStore
    from src.retrieval.embeddings import EmbeddingModel

    embedder = EmbeddingModel(SETTINGS.embedding_model)
    engine = create_engine(SETTINGS.postgres_dsn)
    dimension = embedder.dimension  # triggers the (lazy) model load once, upfront
    return PGVectorStore(engine, dimension=dimension), embedder


def build_pgvector_index(data_dir: str | None = None):
    from src.retrieval.hybrid_pgvector import HybridPGVectorRetriever

    data_dir = data_dir or SETTINGS.data_dir
    documents = load_directory(data_dir)
    if not documents:
        raise RuntimeError(
            f"No supported files (.pdf/.txt/.md) found in '{data_dir}'. "
            "Drop your file there first, e.g. data/AI Engineering.pdf"
        )
    chunks = chunk_documents(documents, chunk_size=SETTINGS.chunk_size, overlap=SETTINGS.chunk_overlap)

    store, embedder = _pgvector_store()
    retriever = HybridPGVectorRetriever(store, embedder=embedder)
    retriever.fit(chunks)  # embeds + upserts into Postgres
    return retriever


def load_pgvector_index():
    from src.retrieval.hybrid_pgvector import HybridPGVectorRetriever

    store, embedder = _pgvector_store()
    return HybridPGVectorRetriever.load_from_store(store, embedder=embedder)


# ---------------------------------------------------------- convenience ---
def load_or_build_index():
    """Used by the API/CLI so a fresh clone 'just works' on first run,
    dispatching to whichever backend SETTINGS.vector_backend selects."""
    if SETTINGS.vector_backend == "postgres":
        try:
            return load_pgvector_index()
        except RuntimeError:
            return build_pgvector_index()
    try:
        return load_index()
    except RuntimeError:
        return build_index()
