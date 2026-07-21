"""
Regression test for bug #3 in docs/bugs-found.md.

build_default_registry() originally only wrapped the ImportError check
around *constructing* RerankedRetriever, but sentence-transformers is
imported lazily inside CrossEncoderReranker._load(), which only runs on
the first real .rerank() call -- deep inside an agent's run(), not at
registry build time. So the crash surfaced mid-query instead of at
startup, and registry.py's try/except never actually caught it.
"""

from __future__ import annotations

import sys

from src.core.interfaces import Document
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, RerankedRetriever


def test_reranker_falls_back_without_crashing_when_dependency_missing(monkeypatch):
    # Forces `import sentence_transformers` to raise ImportError regardless
    # of whether it's actually installed in the environment running this
    # test -- a None entry in sys.modules is the standard way to simulate
    # "this import fails" without needing to actually uninstall anything.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    docs = [Document(id="d1", text="Ravens are highly intelligent birds.", metadata={})]
    base = HybridRetriever()
    base.fit(docs)

    retriever = RerankedRetriever(base=base, reranker=CrossEncoderReranker(), candidate_pool=5)

    # Must NOT raise ModuleNotFoundError -- should gracefully fall back to
    # the un-reranked hybrid candidates instead of crashing mid-query.
    results = retriever.search("intelligent birds", top_k=1)
    assert len(results) == 1
    assert results[0][0].id == "d1"
