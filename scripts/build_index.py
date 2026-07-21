#!/usr/bin/env python
"""Run this after dropping a file into data/, e.g.:

    cp "~/Downloads/AI Engineering.pdf" data/
    python scripts/build_index.py

Builds whichever backend SETTINGS.vector_backend selects:
  memory   (default) -> pickle file at data/.index/hybrid_index.pkl
  postgres            -> upserts into the pgvector table (see
                          docker-compose.yml for a ready-to-run Postgres +
                          pgvector instance, and .env.example for the env
                          vars this needs: VECTOR_BACKEND=postgres,
                          POSTGRES_DSN)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SETTINGS  # noqa: E402
from src.ingestion.indexer import build_index, build_pgvector_index  # noqa: E402


def main() -> None:
    if SETTINGS.vector_backend == "postgres":
        retriever = build_pgvector_index()
        print(f"Indexed {len(retriever.documents)} chunks into Postgres (pgvector) at {SETTINGS.postgres_dsn}")
    else:
        retriever = build_index()
        print(f"Indexed {len(retriever.documents)} chunks. Index saved to data/.index/hybrid_index.pkl")


if __name__ == "__main__":
    main()
