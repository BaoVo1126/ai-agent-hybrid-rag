"""
Reciprocal Rank Fusion (RRF) — extracted as a standalone, reusable function.

Was previously inlined inside HybridRetriever.search() (BM25 + TF-IDF only).
Pulled out here so the new dense-vector retrieval path
(retrieval/pgvector_store.py + retrieval/hybrid_pgvector.py) can fuse
BM25 + pgvector results with the exact same, already-tested fusion logic
instead of duplicating it -- one fusion algorithm, two retrieval backends.

Why rank fusion and not score fusion: BM25 scores, TF-IDF cosine similarity,
and vector cosine/L2 distance all live on different, incomparable scales.
Averaging or summing raw scores from different retrievers is a common but
wrong pattern; RRF sidesteps it entirely by only looking at each result's
*position* in its own ranked list.
"""

from __future__ import annotations

from src.core.interfaces import Document


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """
    ranked_lists: one ranked (Document, score) list per retriever, each
    already sorted best-first. The incoming `score` values are ignored on
    purpose -- only rank position within each list is used.
    """
    rrf_scores: dict[str, float] = {}
    doc_lookup: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked):
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_lookup[doc.id] = doc

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_lookup[doc_id], score) for doc_id, score in fused]
