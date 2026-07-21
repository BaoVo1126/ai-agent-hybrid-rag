"""
Cross-encoder reranker.

Added on top of the existing BM25 + TF-IDF hybrid retrieval (hybrid.py),
which stays completely untouched -- this module only adds a reranking
stage in front of it.

Why a cross-encoder and not just trusting the hybrid RRF ranking: BM25/TFIDF
compare the query and each document independently (bag-of-words overlap),
so two passages that are topically similar but answer a different question
can score close together. A cross-encoder feeds the (query, passage) pair
into the model *together*, which lets it judge relevance far more precisely
-- at the cost of being too slow to run over an entire corpus, which is
exactly why it only reranks the small shortlist HybridRetriever already
narrowed down, instead of replacing it.
"""

from __future__ import annotations

from src.core.interfaces import Document
from src.retrieval.hybrid import HybridRetriever

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Thin wrapper around sentence-transformers' CrossEncoder.

    Lazily loaded: importing this module (or building a registry with
    reranking disabled) never requires the model to be downloaded. Only the
    first call to `.rerank()` triggers the download/load.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # optional dependency, imported lazily

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        if not candidates:
            return []

        try:
            model = self._load()
        except ImportError:
            # Defense in depth: registry.py already checks this before ever
            # constructing a RerankedRetriever, but anyone building one by
            # hand still gets a graceful fallback instead of a crash.
            print(
                "[reranker] sentence-transformers not installed -- returning "
                "un-reranked candidates. Install it with "
                "`pip install sentence-transformers` to enable reranking."
            )
            return candidates[:top_k]
        pairs = [(query, doc.text) for doc, _score in candidates]
        raw_scores = model.predict(pairs)

        reranked = sorted(
            zip((doc for doc, _score in candidates), raw_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(doc, float(score)) for doc, score in reranked[:top_k]]


class RerankedRetriever:
    """
    Drop-in retriever: same `.search(query, top_k)` shape as HybridRetriever,
    so DocumentSearchTool and every existing caller work unmodified whether
    reranking is on or off (see tools/registry.py).
    """

    def __init__(
        self,
        base: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
        candidate_pool: int = 20,
    ) -> None:
        self.base = base
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_pool = candidate_pool

    def fit(self, documents: list[Document]) -> None:
        self.base.fit(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int | None = None,
    ) -> list[tuple[Document, float]]:
        pool = candidate_pool or self.candidate_pool
        # Cast a wide net with the cheap hybrid retriever first, then let the
        # slower, more accurate cross-encoder pick the real top_k out of it.
        candidates = self.base.search(query, top_k=pool, candidate_pool=pool)
        return self.reranker.rerank(query, candidates, top_k=top_k)
