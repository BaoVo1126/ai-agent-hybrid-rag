"""The core RAG tool: lets the agent search whatever document you dropped
into data/. This is intentionally the *only* required tool -- calculator and
summarizer are optional extras that demonstrate multi-tool orchestration."""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Tool
from src.retrieval.hybrid import HybridRetriever


class DocumentSearchTool(Tool):
    name = "document_search"
    description = (
        "Search the ingested document(s) for passages relevant to a query. "
        "Use this whenever the user asks something that could be answered "
        "from the uploaded file (facts, definitions, explanations, quotes)."
    )

    def __init__(self, retriever: HybridRetriever, top_k: int = 4) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query, in the document's language."}
            },
            "required": ["query"],
        }

    def run(self, **kwargs: Any) -> str:
        query = kwargs["query"]
        results = self.retriever.search(query, top_k=self.top_k)
        if not results:
            return "No relevant passages found in the indexed document(s)."

        formatted = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            location = f"{source}" + (f", page {page}" if page else "")
            formatted.append(f"[{location} | score={score:.3f}] {doc.text[:500]}")
        return "\n\n".join(formatted)
