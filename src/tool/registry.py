"""
Tool registry.

A single place that owns "which tools exist right now". Agent strategies
and framework adapters pull tools from here rather than importing individual
tool classes directly, so adding a new tool means registering it once and
it becomes available to every agent strategy and every adapter automatically.
"""

from __future__ import annotations

from src.config import SETTINGS
from src.core.interfaces import Tool
from src.ingestion.indexer import load_or_build_index
from src.tools.calculator_tool import CalculatorTool
from src.tools.rag_tool import DocumentSearchTool
from src.tools.summarize_tool import SummarizeTool


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())


def build_default_registry() -> ToolRegistry:
    """The standard tool set used by the API, CLI, and benchmark."""
    retriever = load_or_build_index()

    if SETTINGS.use_reranker:
        try:
            import sentence_transformers  # noqa: F401  -- cheap import, just checks availability

            from src.retrieval.reranker import CrossEncoderReranker, RerankedRetriever

            retriever = RerankedRetriever(
                base=retriever,
                reranker=CrossEncoderReranker(SETTINGS.reranker_model),
                candidate_pool=SETTINGS.retrieval_candidate_pool,
            )
        except ImportError:
            print(
                "[registry] sentence-transformers not installed -- skipping reranker, "
                "falling back to BM25+TFIDF hybrid retrieval only. "
                "Install it with `pip install sentence-transformers` to enable reranking."
            )

    return ToolRegistry(
        [
            DocumentSearchTool(retriever),
            CalculatorTool(),
            SummarizeTool(),
        ]
    )
