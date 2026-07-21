"""TF-IDF + cosine similarity retriever, numpy-only (brute force), same
baseline as rag-from-scratch. Good at exact term overlap, weak on synonyms --
which is exactly why hybrid.py fuses it with BM25 rather than using it alone."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.core.interfaces import Document
from src.retrieval.bm25 import tokenize


class TFIDFRetriever:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._doc_vectors: np.ndarray | None = None

    def fit(self, documents: list[Document]) -> None:
        self.documents = documents
        tokenized = [tokenize(doc.text) for doc in documents]

        vocab: dict[str, int] = {}
        df_counter: Counter = Counter()
        for tokens in tokenized:
            for term in set(tokens):
                df_counter[term] += 1
            for term in tokens:
                if term not in vocab:
                    vocab[term] = len(vocab)
        self._vocab = vocab

        n_docs = len(documents)
        idf = np.zeros(len(vocab))
        for term, idx in vocab.items():
            idf[idx] = math.log((n_docs + 1) / (df_counter[term] + 1)) + 1
        self._idf = idf

        doc_vectors = np.zeros((n_docs, len(vocab)))
        for i, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            total = len(tokens) or 1
            for term, count in tf.items():
                doc_vectors[i, vocab[term]] = (count / total) * idf[vocab[term]]
        norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._doc_vectors = doc_vectors / norms

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        if self._doc_vectors is None or not self.documents:
            return []
        tokens = tokenize(query)
        query_vec = np.zeros(len(self._vocab))
        tf = Counter(tokens)
        total = len(tokens) or 1
        for term, count in tf.items():
            if term in self._vocab:
                query_vec[self._vocab[term]] = (count / total) * self._idf[self._vocab[term]]
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        sims = self._doc_vectors @ query_vec
        ranked_idx = np.argsort(-sims)[:top_k]
        return [(self.documents[i], float(sims[i])) for i in ranked_idx]
