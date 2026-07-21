"""
Dense embedding model wrapper for the pgvector retrieval path.

Same lazy-import + graceful-fallback pattern as retrieval/reranker.py: this
module can be imported freely without sentence-transformers installed;
only the first `.embed()` call actually needs it, and registry.py checks
availability upfront (mirroring the reranker bug fix from before -- see
docs/bugs-found.md #3) so a missing dependency is caught at startup, not
mid-query.
"""

from __future__ import annotations

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # optional dependency, imported lazily

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        # all-MiniLM-L6-v2 -> 384. Computed from the model rather than
        # hardcoded so swapping DEFAULT_EMBEDDING_MODEL doesn't silently
        # desync the pgvector column width in pgvector_store.py.
        return self._load().get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
