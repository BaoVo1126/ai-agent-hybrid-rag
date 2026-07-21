"""Hybrid BM25 + TF-IDF retrieval via Reciprocal Rank Fusion (RRF), same
fusion strategy as rag-from-scratch: combine *rankings* rather than raw
scores, since BM25 and cosine-similarity scores live on different scales
and are not directly comparable.

RRF itself now lives in retrieval/fusion.py so retrieval/hybrid_pgvector.py
(BM25 + dense vector, see that module) can reuse the exact same fusion
logic instead of a second copy."""

from __future__ import annotations

from src.core.interfaces import Document
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.tfidf import TFIDFRetriever


class HybridRetriever:
    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever()
        self.tfidf = TFIDFRetriever()
        self.documents: list[Document] = []

    def fit(self, documents: list[Document]) -> None:
        self.documents = documents
        self.bm25.fit(documents)
        self.tfidf.fit(documents)

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> list[tuple[Document, float]]:
        bm25_results = self.bm25.search(query, top_k=candidate_pool)
        tfidf_results = self.tfidf.search(query, top_k=candidate_pool)
        return reciprocal_rank_fusion([bm25_results, tfidf_results], top_k=top_k, rrf_k=self.rrf_k)
