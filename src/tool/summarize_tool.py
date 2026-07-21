"""A lightweight extractive summarizer tool. Deliberately NOT another LLM
call -- it demonstrates that not every 'tool' needs to be AI-powered; simple
deterministic text processing is often the right (cheap, fast, testable)
choice inside an agent loop."""

from __future__ import annotations

import re
from typing import Any

from src.core.interfaces import Tool


class SummarizeTool(Tool):
    name = "summarize_text"
    description = (
        "Produce a short extractive summary (top sentences) of a block of text. "
        "Use this to condense long document_search results before answering."
    )

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The text to summarize."}},
            "required": ["text"],
        }

    def run(self, **kwargs: Any) -> str:
        text = kwargs["text"]
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if len(s.split()) > 3]
        if not sentences:
            return text[:300]

        # Rank sentences by term frequency (a classic, dependency-free
        # extractive baseline -- no embeddings needed for a "shrink this text" tool).
        word_counts: dict[str, int] = {}
        for s in sentences:
            for w in re.findall(r"[a-zA-Z]+", s.lower()):
                word_counts[w] = word_counts.get(w, 0) + 1

        def score(sentence: str) -> float:
            words = re.findall(r"[a-zA-Z]+", sentence.lower())
            return sum(word_counts.get(w, 0) for w in words) / (len(words) or 1)

        ranked = sorted(sentences, key=score, reverse=True)[: self.max_sentences]
        ranked_in_order = [s for s in sentences if s in ranked]
        return " ".join(ranked_in_order)
