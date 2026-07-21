"""Shared config. Kept tiny and dependency-free on purpose."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # "mock" (offline, deterministic, no install needed) or "ollama" (free,
    # runs locally via https://ollama.com -- no API key, no cloud account).
    llm_backend: str = os.environ.get("LLM_BACKEND", "mock")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.1")
    request_timeout_seconds: int = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "120"))
    data_dir: str = os.environ.get("DATA_DIR", "data")
    index_dir: str = os.environ.get("INDEX_DIR", "data/.index")
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "60"))
    top_k: int = int(os.environ.get("TOP_K", "4"))

    # Reranking (cross-encoder on top of the BM25+TFIDF hybrid retriever).
    # Off automatically if sentence-transformers isn't installed (see
    # tools/registry.py) so the rest of the system still runs without it.
    use_reranker: bool = os.environ.get("USE_RERANKER", "true").strip().lower() in ("1", "true", "yes")
    reranker_model: str = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    retrieval_candidate_pool: int = int(os.environ.get("RETRIEVAL_CANDIDATE_POOL", "20"))

    # Self-correcting RAG strategy (src/agents/self_correcting_rag_agent.py):
    # how many retrieve->generate->verify loops it may run before returning
    # its best answer flagged as unverified, instead of looping forever.
    self_correction_max_retries: int = int(os.environ.get("SELF_CORRECTION_MAX_RETRIES", "2"))

    # --- Vector storage backend ---
    # "memory": original zero-setup path -- BM25+TFIDF, persisted to a
    #           pickle file (ingestion/indexer.py). Nothing to install, works
    #           on first clone.
    # "postgres": production path -- BM25 (still in-memory, rebuilt at
    #           startup) fused with pgvector dense search
    #           (retrieval/hybrid_pgvector.py). Needs postgres_dsn below and
    #           the `pgvector` Postgres extension + Python package.
    vector_backend: str = os.environ.get("VECTOR_BACKEND", "memory")
    postgres_dsn: str = os.environ.get(
        "POSTGRES_DSN", "postgresql+psycopg2://agentlab:agentlab@localhost:5432/agentlab"
    )
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- Chat session history ---
    # "memory": in-process dict, cleared on restart -- fine for local dev/demo.
    # "postgres": durable storage, survives restarts and works across
    #           multiple API workers/replicas. Read-heavy paths (fetching
    #           recent turns for context) go through Redis first when
    #           REDIS_URL is set; a Redis outage just means slower reads via
    #           Postgres, never a hard failure (see cache/redis_client.py).
    chat_history_backend: str = os.environ.get("CHAT_HISTORY_BACKEND", "memory")
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    session_cache_ttl_seconds: int = int(os.environ.get("SESSION_CACHE_TTL_SECONDS", "3600"))

    @property

    def is_real_mode(self) -> bool:
        """Same dual-mode pattern as before: an explicit switch flips mock -> real.
        Real mode now means "talk to a local Ollama server" instead of a paid API,
        so there's no key to leak and nothing to pay for -- just `LLM_BACKEND=ollama`."""
        return self.llm_backend.strip().lower() == "ollama"


SETTINGS = Settings()
