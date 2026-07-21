from __future__ import annotations

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.tfidf import TFIDFRetriever


def test_bm25_ranks_relevant_doc_first(sample_documents):
    bm25 = BM25Retriever()
    bm25.fit(sample_documents)
    results = bm25.search("reasoning loop agent tools", top_k=3)
    assert results, "BM25 should return at least one result"
    top_doc, _score = results[0]
    assert "agent" in top_doc.text.lower()


def test_tfidf_ranks_relevant_doc_first(sample_documents):
    tfidf = TFIDFRetriever()
    tfidf.fit(sample_documents)
    results = tfidf.search("precision recall reciprocal rank", top_k=3)
    assert results
    top_doc, _score = results[0]
    assert "evaluation" in top_doc.text.lower() or "metrics" in top_doc.text.lower()


def test_bm25_empty_query_does_not_crash(sample_documents):
    bm25 = BM25Retriever()
    bm25.fit(sample_documents)
    results = bm25.search("", top_k=3)
    # No query terms -> zero scores for everything; should not raise.
    assert isinstance(results, list)


def test_bm25_idf_can_be_zero_for_ubiquitous_terms(sample_documents):
    """
    Regression note (mathematical property, not a bug -- learned in
    rag-from-scratch): a term appearing in every document of a small corpus
    yields IDF close to zero, which is expected BM25 behavior, not a defect.
    """
    bm25 = BM25Retriever()
    bm25.fit(sample_documents)
    # None of our terms are literally in all 3 docs, so construct a corpus where one is.
    from src.core.interfaces import Document

    docs = [
        Document(id="a", text="the cat sat"),
        Document(id="b", text="the dog ran"),
        Document(id="c", text="the bird flew"),
    ]
    bm25.fit(docs)
    assert bm25._idf["the"] == 0.0
