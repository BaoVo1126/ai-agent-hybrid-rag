"""
BM25 retriever.

Same formula used in rag-from-scratch. Worth restating the property learned
there: IDF can legitimately go to zero (or negative, clipped to zero here)
when a term appears in most/all documents of a small corpus -- that is
expected BM25 behavior, not a bug, and it's why hybrid.py never relies on
BM25 alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from src.core.interfaces import Document

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self._doc_freqs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._avg_doc_len: float = 0.0
        self._idf: dict[str, float] = {}

    def fit(self, documents: list[Document]) -> None:
        self.documents = documents
        self._doc_freqs = []
        self._doc_lens = []
        df_counter: Counter = Counter()

        for doc in documents:
            tokens = tokenize(doc.text)
            self._doc_lens.append(len(tokens))
            freqs = Counter(tokens)
            self._doc_freqs.append(freqs)
            for term in freqs:
                df_counter[term] += 1

        n_docs = len(documents)
        self._avg_doc_len = sum(self._doc_lens) / n_docs if n_docs else 0.0
        self._idf = {
            term: max(0.0, math.log((n_docs - df + 0.5) / (df + 0.5) + 1e-9))
            for term, df in df_counter.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        if not self.documents:
            return []
        query_terms = tokenize(query)
        scores = [0.0] * len(self.documents)

        for i, freqs in enumerate(self._doc_freqs):
            doc_len = self._doc_lens[i] or 1
            score = 0.0
            for term in query_terms:
                if term not in freqs:
                    continue
                idf = self._idf.get(term, 0.0)
                f = freqs[term]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_doc_len or 1))
                score += idf * (f * (self.k1 + 1)) / (denom or 1e-9)
            scores[i] = score

        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in ranked[:top_k] if score > 0] or ranked[:top_k]
