"""
Default evaluation set.

Ships with a few document-agnostic examples (arithmetic, generic retrieval,
a multi-step task) so `python scripts/run_benchmark.py` works immediately
on a fresh clone with any file dropped into data/. For a meaningful accuracy
number on YOUR document, add a few entries with `expected_keywords` drawn
from facts you know are in your file, e.g.:

    EvalExample(
        query="What year was the transformer architecture introduced?",
        expected_tool="document_search",
        expected_keywords=["2017"],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalExample:
    query: str
    expected_tool: str | None = None  # a tool the agent should call; None = no requirement
    expected_keywords: list[str] = field(default_factory=list)  # any one of these should appear in the final answer


DEFAULT_EVAL_SET: list[EvalExample] = [
    EvalExample(
        query="What is 12 * 8?",
        expected_tool="calculator",
        expected_keywords=["96"],
    ),
    EvalExample(
        query="Summarize the main topic of the ingested document in one sentence.",
        expected_tool="document_search",
        expected_keywords=[],
    ),
    EvalExample(
        query="Explain the key idea discussed in the document and calculate 15 + 27.",
        expected_tool=None,  # this multi-part query should trigger BOTH tools across strategies
        expected_keywords=["42"],
    ),
    EvalExample(
        query="According to the document, what problem is being addressed?",
        expected_tool="document_search",
        expected_keywords=[],
    ),
    EvalExample(
        query="What is (25 + 15) / 2?",
        expected_tool="calculator",
        expected_keywords=["20"],
    ),
]
